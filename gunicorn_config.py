# Gunicorn 配置文件
# 用于生产环境运行 Flask 应用

import multiprocessing
import os

# 服务器配置
bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1  # 根据 CPU 核心数设置工作进程数
worker_class = "sync"
worker_connections = 1000
timeout = 300  # 5分钟超时（训练模型可能需要较长时间）
keepalive = 5

# 日志配置
accesslog = os.path.join(os.path.dirname(__file__), "logs", "access.log")
errorlog = os.path.join(os.path.dirname(__file__), "logs", "error.log")
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 进程命名
proc_name = "quantitative-program"

# 安全配置
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# 性能优化
max_requests = 1000  # 每个工作进程处理请求数后重启
max_requests_jitter = 50
preload_app = True  # 预加载应用，提高性能

