# QuantitativeProgram 服务器部署指南

本指南将帮助你在 Linux 服务器上部署 QuantitativeProgram 项目，并使其能够持久运行。

## 前置要求

- Linux 服务器（Ubuntu/Debian 推荐）
- root 或 sudo 权限
- GitHub 仓库已创建并推送代码

## 快速部署（使用自动脚本）

### 1. 修改部署脚本配置

编辑 `deploy.sh` 文件，修改以下变量：

```bash
GITHUB_REPO_URL="https://github.com/erling-9/Quantitative-Program.git"  # GitHub 仓库地址
DEPLOY_DIR="/opt/quantitative-program"  # 部署目录（可选）
SERVICE_USER="www-data"  # 运行服务的用户（可选）
```

### 2. 运行部署脚本

```bash
# 将脚本上传到服务器，或直接在服务器上克隆仓库
chmod +x deploy.sh
sudo ./deploy.sh
```

## 手动部署步骤

如果自动脚本不适用，可以按照以下步骤手动部署：

### 1. 安装系统依赖

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git nginx
```

### 2. 克隆 GitHub 仓库

```bash
# 创建部署目录
sudo mkdir -p /opt/quantitative-program
cd /opt/quantitative-program

# 克隆仓库
sudo git clone https://github.com/erling-9/Quantitative-Program.git .

# 或者如果已经克隆，更新代码
git pull
```

### 3. 设置 Python 虚拟环境

```bash
cd /opt/quantitative-program

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn  # 生产环境运行 Flask
```

### 4. 创建必要的目录

```bash
mkdir -p uploads logs
```

### 5. 配置 systemd 服务

创建服务文件：

```bash
sudo nano /etc/systemd/system/quantitative-program.service
```

添加以下内容（根据实际情况修改路径和用户）：

```ini
[Unit]
Description=Quantitative Program Trading Signal Server
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/opt/quantitative-program
Environment="PATH=/opt/quantitative-program/venv/bin"
ExecStart=/opt/quantitative-program/venv/bin/gunicorn --bind 0.0.0.0:8000 --workers 4 --timeout 300 --config gunicorn_config.py server:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 6. 启动服务

```bash
# 重新加载 systemd
sudo systemctl daemon-reload

# 启用服务（开机自启）
sudo systemctl enable quantitative-program

# 启动服务
sudo systemctl start quantitative-program

# 检查状态
sudo systemctl status quantitative-program
```

### 7. 配置 Nginx 反向代理（可选但推荐）

#### 7.1 创建 Nginx 配置

```bash
sudo nano /etc/nginx/sites-available/quantitative-program
```

将 `nginx.conf` 文件的内容复制进去，并修改域名。

#### 7.2 启用配置

```bash
sudo ln -s /etc/nginx/sites-available/quantitative-program /etc/nginx/sites-enabled/
sudo nginx -t  # 测试配置
sudo systemctl restart nginx
```

## 常用管理命令

### 服务管理

```bash
# 查看服务状态
sudo systemctl status quantitative-program

# 启动服务
sudo systemctl start quantitative-program

# 停止服务
sudo systemctl stop quantitative-program

# 重启服务
sudo systemctl restart quantitative-program

# 查看日志
sudo journalctl -u quantitative-program -f

# 查看最近 100 行日志
sudo journalctl -u quantitative-program -n 100
```

### 应用日志

```bash
# 访问日志
tail -f /opt/quantitative-program/logs/access.log

# 错误日志
tail -f /opt/quantitative-program/logs/error.log

# 预测日志
tail -f /opt/quantitative-program/logs/prediction_logs.jsonl
```

### 代码更新

```bash
cd /opt/quantitative-program
git pull
sudo systemctl restart quantitative-program
```

## 防火墙配置

如果服务器有防火墙，需要开放端口：

```bash
# UFW (Ubuntu)
sudo ufw allow 8000/tcp
sudo ufw allow 80/tcp   # 如果使用 Nginx
sudo ufw allow 443/tcp  # 如果使用 HTTPS

# firewalld (CentOS/RHEL)
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --permanent --add-port=80/tcp
sudo firewall-cmd --reload
```

## 安全建议

1. **使用 HTTPS**：配置 SSL 证书（Let's Encrypt 免费）
2. **限制访问**：使用防火墙限制只允许特定 IP 访问
3. **定期更新**：保持系统和依赖包更新
4. **备份数据**：定期备份模型文件和日志

## 故障排查

### 服务无法启动

1. 检查日志：`sudo journalctl -u quantitative-program -n 50`
2. 检查 Python 环境：`/opt/quantitative-program/venv/bin/python --version`
3. 检查端口占用：`sudo netstat -tlnp | grep 8000`
4. 手动测试：`cd /opt/quantitative-program && source venv/bin/activate && python server.py`

### 无法访问 API

1. 检查服务状态：`sudo systemctl status quantitative-program`
2. 检查防火墙：`sudo ufw status`
3. 测试本地连接：`curl http://localhost:8000/health`
4. 检查 Nginx 配置：`sudo nginx -t`

### 模型训练失败

1. 检查上传文件大小限制（Nginx 和 Gunicorn）
2. 检查磁盘空间：`df -h`
3. 查看详细错误日志

## 性能优化

1. **调整工作进程数**：根据服务器 CPU 核心数修改 `gunicorn_config.py` 中的 `workers`
2. **使用 Redis**：如果需要缓存，可以集成 Redis
3. **数据库优化**：如果使用数据库，优化查询和索引

## 监控建议

- 使用 `systemctl status` 定期检查服务状态
- 监控日志文件大小，定期清理或轮转
- 设置告警（如使用 Prometheus + Grafana）

## 联系支持

如遇到问题，请检查：
1. 服务日志：`journalctl -u quantitative-program`
2. 应用日志：`logs/error.log`
3. Nginx 日志：`/var/log/nginx/`

