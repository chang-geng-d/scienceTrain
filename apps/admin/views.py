# encoding:utf-8
import os
from flask import *
from server import FLServer as flServ
from client import FederatedClient as flClie
import json

bp = Blueprint("admin", __name__)
flServer:flServ=None
flClients={}

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
                print(str(user['isbad']))
                resp.set_cookie('isBad',str(user['isbad']),max_age=3600)
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
    isBad=request.cookies.get('isBad')=='True'
    return render_template('user.html',username=uName,isBad=isBad)

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

@bp.route('/conn_getLog',methods=['GET','POST'])
def conn_getLog():
    if request.method=='POST':
        data=request.form
        if data['isAdmin']=='1':
            logs={}
            if flServer.serverLog:
                logs['server']=flServer.serverLog.readLog()
            else:
                logs['server']='服务器已关闭'
            for (sid,log) in flServer.userLogs.items():
                logs[sid]=log.readLog()
            return logs
        if flClients[data['sid']].userLog:
            return flClients[data['sid']].userLog.readLog()
        return '错误: 无对应日志或服务器已关闭'

@bp.route('/fl_server/<method>')
def fl_server(method):
    global flServer
    if not flServer:
        flServer=flServ('127.0.0.1',9000)
        flServer.method=method
        print('server 创建')
    return render_template('fl_server.html')

@bp.route('/fl_client/<method>')
def fl_client(method):
    if method in ['none','gan','diffPri']:
        global flClients
        tClie=flClie('127.0.0.1',9000)
        uName=request.cookies.get('uName')
        isBad=request.cookies.get('isBad')
        tClie.isBad=isBad=='True'
        tClie.method=method
        flClients[str(len(flClients))]=tClie
        print(f'client 创建,用户:{uName}\t攻击:{tClie.isBad}')
        return render_template('fl_client.html',sid=len(flClients)-1,isBad=tClie.isBad)
    elif method=='membership':
        # p = subprocess.Popen('python3 ./membership_attack/membership.py', shell=True)
        return render_template('attack.html')
    elif method=='PPA':
        # p = subprocess.Popen('python3 ./PPA_attack/Meta-Classifier/cifar10/Meta-Classifier/train_model.py', shell=True)
        return render_template('ppa.html')
    elif method=='homomorphic':
        # p = subprocess.Popen('python3 ./Homomorphic_Encryption/Paillier/encrypt.py', shell=True)
        return render_template('homomorphic.html')

@bp.route('/conn_manageFl',methods=['GET','POST'])
def conn_manageFl():
    if request.method=='POST':
        data=request.form
        if data['isAdmin']=='1':
            global flServer
            if data['method']=='start':
                print(flServer)
                flServer.start()
                print('server start')
            elif data['method']=='stop':
                print('enter stop')
                flServer.stop()
                flServer.delLog(False)
                flServer=None
                print('server stop')
            # elif data['method']=='delete':
            #     print('enter delete')
            #     flServer.stop()
            #     flServer.delLog(False)
            #     flServer=None
            #     print('server deleted')
        else:
            global flClients
            if data['method']=='start':
                print('client start')
                flClients[data['sid']].run()
                print(f"client start sid:{data['sid']}")
                while not flClients[data['sid']].sid:   #阻塞方式,网络不好时可能出问题
                    pass
                return str(flClients[data['sid']].sid),200
            elif data['method']=='stop':
                print('enter client stop')
                flClients[data['sid']].stop()
                flClients.pop(data['sid'])
                print('client stop')
            # elif data['method']=='delete':
            #     flClients[data['sid']].stop()
            #     flClients.pop(data['sid'])
            #     print('client deleted')
        return '',200

@bp.route('/result')
def result():
    with open('membership_attack/log.txt', encoding='utf-8') as file_obj:
        contents = file_obj.read()
        print(contents.rstrip())
    return render_template('result.html',result = contents.rstrip())

@bp.route('/mem_graph')
def mem_graph():
    return render_template('mem_graph.html')

@bp.route('/ppa_result')
def ppa_result():
    with open('PPA_attack/Meta-Classifier/cifar10/Meta-Classifier/log.txt', encoding='utf-8') as file_obj:
        contents = file_obj.read()
        print(contents.rstrip())
    return render_template('ppa_result.html',result = contents.rstrip())

@bp.route('/ppa_graph')
def ppa_graph():
    return render_template('ppa_graph.html')

# @bp.route('/GAN')
# def gan():
#     # p = subprocess.Popen('python3 ./gan_attack/gan.py', shell=True)
#     return render_template('gan.html')

# @bp.route('/gan_result')
# def gan_result():
#     with open('gan_attack/log.txt', encoding='utf-8') as file_obj:
#         contents = file_obj.read()
#         print(contents.rstrip())
#     return render_template('gan_result.html',result = contents.rstrip())

@bp.route('/fl_result/gan')
def gan_graph():
    picPaths=os.listdir('static/images/gan_imgs')
    picPaths.sort(key=lambda x:int(x.split('.')[0]))
    return render_template('gan_graph.html',picPaths=picPaths)

@bp.route('/homo_result')
def homo_result():
    with open('Homomorphic_Encryption/Paillier/log.txt', encoding='utf-8') as file_obj:
        contents = file_obj.read()
        print(contents.rstrip())
    return render_template('homo_result.html',result = contents.rstrip())