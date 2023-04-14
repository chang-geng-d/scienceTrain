# encoding:utf-8
import logging
from flask import Flask
from apps.admin.views import bp

DEBUG = False
log = logging.getLogger('werkzeug')
log.disabled = True

def create_app():
    app = Flask(__name__)
    # 注册蓝图
    app.secret_key="fanfneaoidamfoiaf"
    app.register_blueprint(bp)
    app.config.from_object('config')
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='127.0.0.1', port=8000, debug=DEBUG)   # 开了debug后flserver服务器无法启动，注意
