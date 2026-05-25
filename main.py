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
from pandas.io.sas.sasreader import FilePath
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

os.makedirs(app.config['UPLOAD_FOLDER'],exixt_ok=True)

current_df =None
current_filename=None

#页面路由
@app.route('/')
def index():
    return render_template('index.html')

# 文件的上传与导出
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
@app.route('/upload',methods=['POST'])
def upload_file():
    # 全局变量DataFrame二维表格与文件名
    global current_df,currnet_filename
    try:
        file=request.files.get('file')
        if not file or file.filename=='':
            return jsonify({'error':'未上传文件！'}),400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({
                'error': f'不支持的文件类型。请上传 {", ".join(ALLOWED_EXTENSIONS)} 格式的文件'
            }), 400

        currnet_filename=file.filename
        filepath=os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
        file.save(filepath)

        # 读取，因为数据集来源于Kaggle，所以编码不做对中文的适配
        if file.filename.lower().endswith('.csv'):
            # 如果是CSV文件
            current_df=pd.read_csv(filepath,encoding='utf-8')
        else:
            # XLSX与XLS
            current_df=pd.read_excel(filepath)

        # 数据库相关，暂时先不用
        # save_history('upload')

        # 预览
        preview_col=20  # 可预览行数
        preview=current_df.head(preview_col).fillna('').to_dict(orient='records')
        return jsonify({
            'columns':current_df.columns.tolist(),
            'preview':preview,
            'shape':[len(current_df),len(current_df.columns)]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error':f'文件上传失败：{str(e)}'}),500

@app.route('/export',methods=['GET'])
def export_data():
    global current_df
    if current_df is None:
        return jsonify({'error': '无数据可导出'}), 400

    output=io.StringIO()
    current_df.to_csv(output,index=False,encoding='utf-8-sig')
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

@app.route('/clean',methods=['POST'])
def clean_data():
    global current_df
    try:
        data=request.get_json() or {}
        method=data.get('method','median')
        z_threshold=float(data.get('z_threshold',3))

        df=get_data()
        initial_rows=len(df)
        missing_before=df.isnull().sum().sum()

        # 处理缺失值
        if method == 'drop':
            df=df.dropna()
        # 均值填充
        elif method=='mean':
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(),inplace=True)
                else:
                    mode_vals=df[col].mode()
                    if not mode_vals.empty:
                        df[col].fillna(mode_vals[0],inplace=True)
        # 默认中位数填充
        else:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    mode_vals = df[col].mode()
                    if not mode_vals.empty:
                        df[col].fillna(mode_vals[0], inplace=True)

        # Z-score 异常值检测
        numeric_cols=df.select_dtypes(include='number').columns
        outlier_count=0
        for col in numeric_cols:
            col_mean = df[col].mean()
            col_std=df[col].std()
            if col_std>0:
                z_scores=np.abs((df[col]-col_mean)/col_std)
                outlier_count+=(z_scores>z_threshold).sum()

        current_df=df
        missing_after=df.isnull().sum().sum()
        filled_count=missing_before-missing_after

        message=f"清洗完成：填充了 {filled_count} 个缺失值，检测到 {outlier_count} 个可能的异常值（Z-score > {z_threshold}）"
        # 数据库处理，暂时不用
        # save_history('clean',message)

        # 可预览行数
        preview_len=20
        preview = current_df.head(preview_len).fillna('').to_dict(orient='records')
        return jsonify({
            'message': message,
            'preview': preview
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


    # 图表数据
    @app.route('/chart',methods=['GET'])
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
                    chart_points.append([float(row[x_col]), float(row[y_col])])
                except (ValueError, TypeError):
                    continue

            return jsonify({'data': chart_points})
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500



    # 聚类分析，对比 K‑Means 和 DBSCAN 算法，利用轮廓系数衡量各自的效果
    @app.route('/cluster',methods=['POST'])
    def cluster_analysis():
        global current_df
        try:
            df=get_data()
            # 选择数据列
            numeric_cols=df.select_dtypes(include='number').columns.tolist()
            if len(numeric_cols)<2:
                return jsonify({'error': '至少需要2个数值列'}), 400

            #数据标准化
            num_data=df[numeric_cols].dropna()
            scaler = StandardScaler()
            scaled = scaler.fit_transform(num_data)

            plot_x=scaled[:,0].tolist()
            plot_y=scaled[:,1].tolist()
            
            results={'scores':{},'kmeans':[],'dbscan':[]}

            #K-Means
            # 暂定5个簇，数据量控制在1000~2000行之间
            kmeans=KMeans(n_clusters=5,random_state=42,n_init=10)
            kmeans_labels=kmeans.fit_predict(scaled)
            #轮廓系数评估
            results['scores']['K-Means']=round(silhouette_score(scaled,kmeans_labels),4)
            results['kmeans']=[[float(plot_x[i]),float(plot_y[i])]for i in range(len(plot_x))]



            #DBSCAN
            try:
                # 邻域半径 最小邻居数
                dbscan=DBSCAN(eps=0.5,min_samples=5)
                dbscan_labels=descan.fit_predict(scaled)
                unique_labels=set(dbscan_labels)
                if len(unique_labels)>1 and -1 not in unique_labels:
                    score = silhouette_score(scaled, dbscan_labels)
                elif len(unique_labels)>1:
                    # 有异常值，剔除，在有效值上计算
                    mask= dbscan_labels !=-1
                    if mask.sum()>1and len(set(dbscan_labels[mask]))>1:
                        score=silhouette_score(scaled[mask],dbscan_labels[mask])
                    else:
                        score=0.0
                else:
                    score=0.0
                # 同样的，轮廓系数评估
                results['scores']['DBSCAN'] = round(score, 4)
                results['dbscan'] = [[float(plot_x[i]), float(plot_y[i])] for i in range(len(plot_x))]
            except Exception as e:
                results['scores']['DBSCAN'] = 0.0
                results['dbscan'] = [[float(plot_x[i]), float(plot_y[i])] for i in range(len(plot_x))]

            #将聚类标签加入数据
            current_df['cluster_kmeans']=kmeans_labels
            # 数据库历史导入
            # save_history('clustering', f"K-Means:{results['scores']['K-Means']}, DBSCAN:{results['scores']['DBSCAN']}")

            return jsonify(results)
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500


if __name__=="__main__":
    app.run(debug=True)
