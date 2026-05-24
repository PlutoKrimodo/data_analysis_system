import io
import os
from re import S
import sqlite3
import traceback
from datetime import datetime

# 数据处理
import numpy as np
from numpy._core import numeric
import pandas as pd

# WEB框架
from flask import Flask, request, render_template, jsonify, send_file, session

# 机器学习
from pandas.api.types import is_numeric_dtype
from pandas.io.sas.sasreader import FilePath
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from app1 import save_history

app=Flask(__name__)
#密钥，因为不是实际开发，所以用了固定值
app.config['SECRET_KEY']='DAS'
# 最大允许上传32MB的文件
app.config['MAX_CONTENT_LENGTH']= 32*1024*1024
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
            return jsonify({'错误':'未上传文件！'}),400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({
                '错误': f'不支持的文件类型。请上传 {", ".join(ALLOWED_EXTENSIONS)} 格式的文件'
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
        preview_col=60  # 可预览行数
        preview=current_df.head(preview_col).fillna('').to_dict(orient='records')
        return jsonify({
            'columns':current_df.columns.tolist(),
            'preview':preview,
            'shape':[len(current_df),len(current_df.columns)]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'错误':f'文件上传失败：{str(e)}'}),500

@app.route('/export',methods=['GET'])
def export_data():
    global current_df
    if current_df is None:
        return jsonify({'错误': '无数据可导出'}), 400

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
                    if not mode_vals.empty():
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
        missing_after=df.isnull().num().num()
        filled_count=missing_before-missing_after

        message=f"清洗完成：填充了 {filled_count} 个缺失值，检测到 {outlier_count} 个可能的异常值（Z-score > {z_threshold}）"
        # 数据库处理，暂时不用
        save_history('clean',message)

        preview_len=60
        preview = current_df.head(preview_len).fillna('').to_dict(orient='records')
        return jsonify({
            'message': message,
            'preview': preview
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'错误': str(e)}), 500


if __name__=="__main__":
    app.run(debug=True)