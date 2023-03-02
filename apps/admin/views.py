# encoding:utf-8
from flask import Blueprint, render_template, request, redirect,flash,make_response
import random
import time
import subprocess
import os
import server as s
import client as c
import json

bp = Blueprint("admin", __name__)


@bp.route('/')
def _login():
    return redirect('/login')


@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    username=request.form.get('username')
    pwd=request.form.get('password')
    with open('configs/users.json','r') as f:
        users=json.load(f)
    for user in users:
        if user['username']==username:
            if user['password']==pwd:
                if user['isAdmin']:
                    resp=make_response(redirect('/manage'))
                else:
                    resp=make_response(redirect('/user'))
                resp.set_cookie('uName',username,max_age=3600)
                return resp
            else:
                break
    flash('不存在的账户或错误的密码')
    return render_template('login.html')

@bp.route('/manage')
def manage():
    uName=request.cookies.get('uName')
    with open('configs/log.json','r') as f:
        dics=json.load(f)
    with open('configs/users.json','r') as f:
        userlist=json.load(f)
    return render_template('manage.html',username=uName,dics=dics,userlist=userlist)

@bp.route('/user')
def user():
    uName=request.cookies.get('uName')
    return render_template('user.html',username=uName)

@bp.route('/conn_chgUser',methods=['POST'])
def conn_chgUser():
    if request.method=='POST':
        data=request.form
        if data['uName']=='' or data['uPass']=='':
            return 'Precondition Failed',412
        with open('configs/users.json','r') as f:
            ulist=json.load(f)
        if (data['method']=="insert"):
            ulist.append({
                'id':str(int(ulist[-1]['id'])+1),
                'username':data['uName'],'password':data['uPass'],
                'isAdmin':False,'isbad':False})
        elif(data['method']=="delete"):
            for i in range(len(ulist)):
                if ulist[i]['username']==data['uName']:
                    del ulist[i]
                    break
        elif(data['method']=="update"):
            for i in range(len(ulist)):
                if ulist[i]['username']==data['uName']:
                    ulist[i]['password']=data['uPass']
                    break
        with open('configs/users.json','w') as f:
            json.dump(ulist,f)
    return 'ok',200

@bp.route('/attack')
def attack():
    # p = subprocess.Popen('python3 ./membership_attack/membership.py', shell=True)
    return render_template('attack.html')

@bp.route('/result')
def result():
    with open('membership_attack/log.txt', encoding='utf-8') as file_obj:
        contents = file_obj.read()
        print(contents.rstrip())
    return render_template('result.html',result = contents.rstrip())

@bp.route('/mem_graph')
def mem_graph():
    return render_template('mem_graph.html')

@bp.route('/PPA')
def ppa():
    # p = subprocess.Popen('python3 ./PPA_attack/Meta-Classifier/cifar10/Meta-Classifier/train_model.py', shell=True)
    return render_template('ppa.html')

@bp.route('/ppa_result')
def ppa_result():
    with open('PPA_attack/Meta-Classifier/cifar10/Meta-Classifier/log.txt', encoding='utf-8') as file_obj:
        contents = file_obj.read()
        print(contents.rstrip())
    return render_template('ppa_result.html',result = contents.rstrip())

@bp.route('/ppa_graph')
def ppa_graph():
    return render_template('ppa_graph.html')

@bp.route('/GAN')
def gan():
    # p = subprocess.Popen('python3 ./gan_attack/gan.py', shell=True)
    return render_template('gan.html')

@bp.route('/gan_result')
def gan_result():
    with open('gan_attack/log.txt', encoding='utf-8') as file_obj:
        contents = file_obj.read()
        print(contents.rstrip())
    return render_template('gan_result.html',result = contents.rstrip())

@bp.route('/gan_graph')
def gan_graph():
    return render_template('gan_graph.html')

@bp.route('/homomorphic')
def homomorphic():
    # p = subprocess.Popen('python3 ./Homomorphic_Encryption/Paillier/encrypt.py', shell=True)
    return render_template('homomorphic.html')

@bp.route('/homo_result')
def homo_result():
    with open('Homomorphic_Encryption/Paillier/log.txt', encoding='utf-8') as file_obj:
        contents = file_obj.read()
        print(contents.rstrip())
    return render_template('homo_result.html',result = contents.rstrip())