import io
import os
from re import S

#数据库 用户信息登记
import sqlite3
import traceback

# 时间 供历史记录使用
from datetime import datetime

# 数据处理
import numpy as np
from numpy._core import numeric
import pandas as pd

# WEB框架
from flask import Flask, request, render_template, jsonify, send_file, session

# 数据分析
from pandas.api.types import is_numeric_dtype
from sklearn.cluster import KMeans, DBSCAN, k_means
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score



app=Flask(__name__)
#密钥，因为不是实际开发，所以用了固定值
app.config['SECRET_KEY']='DAS'
# 最大允许上传16MB的文件
app.config['MAX_CONTENT_LENGTH']= 16*1024*1024
# 上传的文件存放文件夹
app.config['UPLOAD_FOLDER']='uploads'

os.makedirs(app.config['UPLOAD_FOLDER'],exist_ok=True)

current_df =None
current_filename=None

# 存储用户信息的数据库
def init_db():
    conn=sqlite3.connect('database.db')
    c=conn.cursor()
    # id username password
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')

    # 历史记录 
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  filename TEXT,
                  analysis_type TEXT,
                  result TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

init_db()

# 历史记录
def save_history(analysis_type,result=""):
    if 'user_id' in session:
        conn=sqlite3.connect('database.db')
        conn.execute(
            "INSERT INTO history (user_id, filename, analysis_type, result) VALUES (?, ?, ?, ?)",
            (session['user_id'], current_filename or 'unknown', analysis_type, result)
        )
        conn.commit()
        conn.close()


#页面路由
@app.route('/')
def index():
    return render_template('index.html')

# 文件的上传与导出
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
@app.route('/upload',methods=['POST'])
def upload_file():
    # 全局变量DataFrame二维表格与文件名
    global current_df, current_filename   # 修正拼写
    try:
        file=request.files.get('file')
        if not file or file.filename=='':
            return jsonify({'error':'未上传文件！'}),400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({
                'error': f'不支持的文件类型。请上传 {", ".join(ALLOWED_EXTENSIONS)} 格式的文件'
            }), 400

        current_filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], current_filename)
        file.save(filepath)

        # 读取，因为数据集来源于Kaggle，所以编码不做对中文的适配
        if file.filename.lower().endswith('.csv'):
            # 如果是CSV文件
            current_df = pd.read_csv(filepath, encoding='utf-8')
        else:
            # XLSX与XLS
            current_df = pd.read_excel(filepath)

        # 历史记录
        save_history('upload')

        # 预览
        preview_col=20  # 可预览行数
        preview = current_df.head(preview_col).fillna('').to_dict(orient='records')
        return jsonify({
            'columns': current_df.columns.tolist(),
            'preview': preview,
            'shape': [len(current_df), len(current_df.columns)]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'文件上传失败：{str(e)}'}), 500

@app.route('/export', methods=['GET'])
def export_data():
    global current_df
    if current_df is None:
        return jsonify({'error': '无数据可导出'}), 400

    output = io.StringIO()
    current_df.to_csv(output, index=False, encoding='utf-8-sig')
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8-sig')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='processed_data.csv'
    )

# 数据清洗

# 获取数据
def get_data():
    global current_df
    if current_df is None:
        raise ValueError("无数据，请先上传文件")
    return current_df

@app.route('/clean', methods=['POST'])
def clean_data():
    global current_df
    try:
        data = request.get_json() or {}
        method = data.get('method', 'median')
        z_threshold = float(data.get('z_threshold', 3))

        df = get_data()
        initial_rows = len(df)
        missing_before = df.isnull().sum().sum()

        drop_flag=False

        # 处理缺失值
        if method == 'drop':
            df = df.dropna()
            drop_flag=True
        # 均值或者中位数填充
        else:
            fill_type='mean' if method=='mean' else'median'
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if fill_type=='mean':
                        stat=df[col].mean()
                    else:
                        stat=df[col].median()
                    
                    if pd.isna(stat):
                        stat=0

                    df[col]=df[col].fillna(stat)
                else:
                    # 非数值列直接填入众数
                    mode_vals=df[col].mode()
                    if not mode_vals.empty:
                        df[col]=df[col].fillna(mode_vals[0])
        

        # Z-score 异常值检测
        numeric_cols = df.select_dtypes(include='number').columns
        outlier_count = 0
        for col in numeric_cols:
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std > 0:
                z_scores = np.abs((df[col] - col_mean) / col_std)
                outlier_count += (z_scores > z_threshold).sum()

        current_df = df
        missing_after = df.isnull().sum().sum()
        filled_count = missing_before - missing_after

        if drop_flag==False:
            message = f"清洗完成：填充了 {filled_count} 个缺失值，检测到 {outlier_count} 个可能的异常值（Z-score > {z_threshold}）"
        elif drop_flag==True:
            message = f"清洗完成：删除了 {filled_count} 行包含缺失值的数据，检测到 {outlier_count} 个可能的异常值（Z-score > {z_threshold}）"
        # 历史记录
        save_history('clean',message)

        # 可预览行数
        preview_len = 20
        preview = current_df.head(preview_len).fillna('').to_dict(orient='records')

        return jsonify({
            'message': message,
            'preview': preview
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# 图表数据
@app.route('/chart', methods=['GET'])
def chart_data():
    try:
        # 使用者自定义列表内的 “列名” 作为x,y轴信息
        x_col = request.args.get('x')
        y_col = request.args.get('y')
        df = get_data()

        if x_col not in df.columns or y_col not in df.columns:
            return jsonify({'error': '列名不存在'}), 400

        clean = df[[x_col, y_col]].dropna()

        chart_points = []
        for _, row in clean.iterrows():
            try:
                chart_points.append([row[x_col], float(row[y_col])])
            except (ValueError, TypeError):
                continue

        return jsonify({'data': chart_points})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# 聚类分析，对比 K‑Means 和 DBSCAN 算法，利用轮廓系数衡量各自的效果
@app.route('/cluster', methods=['POST'])
def cluster_analysis():
    global current_df
    try:
        df = get_data()
        # 选择数据列（只使用原始数值列）
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        # 排除可能残留的聚类标签列
        if 'cluster_kmeans' in numeric_cols:
            numeric_cols.remove('cluster_kmeans')
        if len(numeric_cols) < 2:
            return jsonify({'error': '至少需要2个数值列'}), 400

        #数据标准化
        num_data = df[numeric_cols].dropna()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(num_data)

        plot_x = scaled[:,0].tolist()
        plot_y = scaled[:,1].tolist()
        
        results = {'scores': {}, 'kmeans': [], 'dbscan': []}

        #K-Means
        # 暂定5个簇，数据量控制在1000~2000行之间
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(scaled)
        #轮廓系数评估
        results['scores']['K-Means'] = round(silhouette_score(scaled, kmeans_labels), 4)
        results['kmeans'] = [[float(plot_x[i]), float(plot_y[i])] for i in range(len(plot_x))]
        results['kmeans_labels'] = kmeans_labels.tolist()  # 将 numpy 数组转为列表



        #DBSCAN
        try:
            # 邻域半径 最小邻居数
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(scaled)
            unique_labels = set(dbscan_labels)
            if len(unique_labels) > 1 and -1 not in unique_labels:
                score = silhouette_score(scaled, dbscan_labels)
            elif len(unique_labels) > 1:
                # 有异常值，剔除，在有效值上计算
                mask = dbscan_labels != -1
                if mask.sum() > 1 and len(set(dbscan_labels[mask])) > 1:
                    score = silhouette_score(scaled[mask], dbscan_labels[mask])
                else:
                    score = 0.0
            else:
                score = 0.0
            # 同样的，轮廓系数评估
            results['scores']['DBSCAN'] = round(score, 4)
            results['dbscan'] = [[float(plot_x[i]), float(plot_y[i])] for i in range(len(plot_x))]
            results['dbscan_labels'] = dbscan_labels.tolist()  # 新增
        except Exception as e:
            results['scores']['DBSCAN'] = 0.0
            results['dbscan'] = [[float(plot_x[i]), float(plot_y[i])] for i in range(len(plot_x))]


        # 历史存储
        save_history('clustering', f"K-Means:{results['scores']['K-Means']}, DBSCAN:{results['scores']['DBSCAN']}")

        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500




# 注册 登录与历史记录
@app.route('/register', methods=['POST'])
def register():
    try:
        data= request.get_json()
        # 读取输入的用户名，密码
        username=data.get('username','').strip()
        password=data.get('password','').strip()

        if not username or not password:
            return jsonify({'status': 'error', 'message': '用户名和密码不能为空'})
        
        conn=sqlite3.connect('database.db')
        try:
            # 存入数据库
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            return jsonify({'status': 'success'})
        except sqlite3.IntegrityError:
            return jsonify({'status': 'error', 'message': '用户名已存在'})
        finally:
            conn.close()
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()

        conn = sqlite3.connect('database.db')
        user = conn.execute(
            "SELECT id, username FROM users WHERE username=? AND password=?",
            (username, password)
        ).fetchone()
        conn.close()

        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': '用户名或密码错误'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/history', methods=['GET'])
def get_history():
    if 'user_id' not in session:
        return jsonify({'error': '未登录！查看历史记录需要登录！'})

    conn = sqlite3.connect('database.db')
    records = conn.execute(
        "SELECT filename, analysis_type, result, timestamp FROM history WHERE user_id=? ORDER BY timestamp DESC LIMIT 20",
        (session['user_id'],)
    ).fetchall()
    conn.close()

    return jsonify([
        {'filename': r[0], 'type': r[1], 'time': r[3]}
        for r in records
    ])

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'status': 'success'})

if __name__ == "__main__":
    app.run(debug=True)
