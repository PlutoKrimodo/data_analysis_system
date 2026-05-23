import io
import base64
import traceback

import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_file
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn import linear_model
from wordcloud import WordCloud, STOPWORDS

# 以下导入用于保持与原 main.py 风格一致（实际 Web 中不强制使用 matplotlib/seaborn）
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16MB

# 全局变量存储当前 DataFrame
current_df = None

# ------------------- 辅助函数（原 main.py 中的样式设置，保留备用） -------------------
def set_seaborn_properties(context='talk', font_scale=0.8):
    sns.set_theme(context=context, font='STXIHEI', font_scale=font_scale,
                  rc={'axes.unicode_minus': False,
                      'figure.figsize': (12, 8),
                      'figure.dpi': 150})

def get_data():
    """安全获取当前数据，若无数据则抛出异常"""
    if current_df is None:
        raise ValueError("无数据，请先上传文件")
    return current_df

# ------------------- 基础路由 -------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_df
    try:
        file = request.files['file']
        if not file:
            return jsonify({'error': '未选择文件'}), 400
        
        filename = file.filename.lower()
        if filename.endswith('.csv'):
            try:
                current_df = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                current_df = pd.read_csv(file, encoding='gbk')
        elif filename.endswith(('.xlsx', '.xls')):
            current_df = pd.read_excel(file)
        else:
            return jsonify({'error': '仅支持 CSV 或 Excel 文件'}), 400
        
        preview = current_df.head(10).fillna('').to_dict(orient='records')
        return jsonify({
            'columns': current_df.columns.tolist(),
            'shape': current_df.shape,
            'preview': preview
        })
    except Exception as e:
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

@app.route('/export')
def export_data():
    if current_df is None:
        return jsonify({'error': '无数据'}), 400
    output = io.StringIO()
    current_df.to_csv(output, index=False, encoding='utf-8-sig')
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8-sig')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='analyzed_data.csv'
    )

@app.route('/clean/missing', methods=['POST'])
def clean_missing():
    global current_df
    try:
        data = request.get_json()
        strategy = data.get('strategy', 'drop')
        df = get_data()
        
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'fill_mean':
            numeric_cols = df.select_dtypes(include='number').columns
            for col in numeric_cols:
                df[col].fillna(df[col].mean(), inplace=True)
        else:
            return jsonify({'error': '不支持的处理策略'}), 400
        
        current_df = df
        return jsonify({
            'message': f'缺失值处理完成（{strategy}）',
            'new_shape': current_df.shape
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect/outliers', methods=['GET'])
def detect_outliers():
    try:
        col = request.args.get('column')
        df = get_data()
        if col not in df.columns:
            return jsonify({'error': f'列 "{col}" 不存在'}), 400
        if not pd.api.types.is_numeric_dtype(df[col]):
            return jsonify({'error': '只能检测数值列'}), 400
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        
        return jsonify({
            'outlier_count': len(outliers),
            'outlier_values': outliers[col].head(20).tolist(),
            'lower_bound': lower,
            'upper_bound': upper
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/cluster', methods=['POST'])
def cluster_analysis():
    global current_df
    try:
        data = request.get_json()
        n_clusters = data.get('n_clusters', 3)
        df = get_data()
        
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) < 2:
            return jsonify({'error': '至少需要2个数值列才能进行聚类'}), 400
        
        X = df[numeric_cols].copy()
        X.fillna(X.mean(), inplace=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        current_df['cluster'] = labels
        
        cluster_means = current_df.groupby('cluster')[numeric_cols].mean().to_dict(orient='index')
        scatter_col1 = numeric_cols[0]
        scatter_col2 = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        scatter_data = [
            [float(row[scatter_col1]), float(row[scatter_col2]), int(row['cluster'])]
            for _, row in current_df.iterrows()
        ]
        
        return jsonify({
            'labels': labels.tolist(),
            'cluster_means': cluster_means,
            'scatter_data': scatter_data,
            'x_col': scatter_col1,
            'y_col': scatter_col2,
            'numeric_columns': numeric_cols
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ------------------- 高级分析路由 -------------------
@app.route('/analyze/regression', methods=['POST'])
def regression_analysis():
    """对当前数据中的两个数值列进行线性回归拟合，返回回归系数和拟合线数据点"""
    try:
        req = request.get_json()
        x_col = req.get('x_col')
        y_col = req.get('y_col')
        df = get_data()
        
        if x_col not in df.columns or y_col not in df.columns:
            return jsonify({'error': '列名不存在'}), 400
        
        # 去除缺失值
        clean = df[[x_col, y_col]].dropna()
        if len(clean) < 2:
            return jsonify({'error': '有效数据点不足'}), 400
        
        X = clean[[x_col]].values
        y = clean[y_col].values
        model = linear_model.LinearRegression()
        model.fit(X, y)
        
        # 生成拟合线
        x_range = np.linspace(X.min(), X.max(), 50)
        y_pred = model.predict(x_range.reshape(-1, 1))
        regression_line = [{'x': float(x_range[i]), 'y': float(y_pred[i])} for i in range(len(x_range))]
        original_points = [{'x': float(X[i][0]), 'y': float(y[i])} for i in range(len(X))]
        
        return jsonify({
            'slope': model.coef_[0],
            'intercept': model.intercept_,
            'r2': model.score(X, y),
            'regression_line': regression_line,
            'original_points': original_points
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/polynomial_predict', methods=['POST'])
def polynomial_predict():
    """
    对当前数据中的时间列和目标列进行多项式回归（3次）并预测未来
    请求JSON: {"time_col": "Year", "target_col": "Value", "future_steps": 10}
    """
    try:
        req = request.get_json()
        time_col = req.get('time_col')
        target_col = req.get('target_col')
        future_steps = int(req.get('future_steps', 10))
        df = get_data()
        
        if time_col not in df.columns or target_col not in df.columns:
            return jsonify({'error': '列名不存在'}), 400
        
        clean = df[[time_col, target_col]].dropna()
        if len(clean) < 4:
            return jsonify({'error': '数据点不足，至少需要4个点进行3次多项式拟合'}), 400
        
        X = clean[time_col].values.reshape(-1, 1)
        y = clean[target_col].values
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        model = linear_model.LinearRegression()
        model.fit(X_poly, y)
        
        # 历史数据点
        historical = [{'year': int(X[i][0]), 'value': float(y[i])} for i in range(len(X))]
        
        # 预测未来
        last_year = X[-1][0]
        future_years = np.arange(last_year + 1, last_year + future_steps + 1).reshape(-1, 1)
        future_poly = poly.transform(future_years)
        future_pred = model.predict(future_poly)
        future = [{'year': int(future_years[i][0]), 'value': float(future_pred[i])} for i in range(len(future_years))]
        
        return jsonify({
            'historical': historical,
            'future': future,
            'coefficients': model.coef_.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/wordcloud', methods=['GET'])
def generate_wordcloud():
    """
    对当前数据中的某一列（文本或分类列）生成词云，返回 base64 图片
    请求参数: ?column=列名
    """
    try:
        col = request.args.get('column')
        df = get_data()
        if col not in df.columns:
            return jsonify({'error': '列不存在'}), 400
        
        # 将列中所有非空值转为字符串
        text_series = df[col].dropna().astype(str)
        if text_series.empty:
            return jsonify({'error': '该列没有有效文本'}), 400
        
        text = ' '.join(text_series)
        # 生成词云
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              max_words=100, stopwords=STOPWORDS).generate(text)
        
        # 转为 base64
        img = io.BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()
        return jsonify({'image_base64': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ------------------- 启动 -------------------
if __name__ == '__main__':
    app.run(debug=True)