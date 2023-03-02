# encoding:utf-8
from flask import Flask
from apps.admin import bp as admin_bp

DEBUG = True

def create_app():
    app = Flask(__name__)
    # 注册蓝图
    app.secret_key="fanfneaoidamfoiaf"
    app.register_blueprint(admin_bp)
    app.config.from_object('config')
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='127.0.0.1', port=8000, debug=True)
