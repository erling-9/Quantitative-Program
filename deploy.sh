#!/bin/bash
# QuantitativeProgram 服务器部署脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始部署 QuantitativeProgram"
echo "=========================================="

# 配置变量（请根据实际情况修改）
GITHUB_REPO_URL="https://github.com/erling-9/Quantitative-Program.git"  # GitHub 仓库地址
DEPLOY_DIR="/opt/quantitative-program"  # 部署目录
SERVICE_USER="www-data"  # 运行服务的用户（可选，如果不需要特定用户可以注释掉）

# 检查是否为 root 用户
if [ "$EUID" -ne 0 ]; then 
    echo "请使用 sudo 运行此脚本"
    exit 1
fi

# 1. 更新系统包
echo "更新系统包..."
apt-get update

# 2. 安装必要的系统依赖
echo "安装系统依赖..."
apt-get install -y python3 python3-pip python3-venv git nginx

# 3. 创建部署目录
echo "创建部署目录: $DEPLOY_DIR"
mkdir -p $DEPLOY_DIR
cd $DEPLOY_DIR

# 4. 克隆或更新 GitHub 仓库
if [ -d ".git" ]; then
    echo "更新代码..."
    git pull
else
    echo "克隆代码仓库..."
    git clone $GITHUB_REPO_URL .
fi

# 5. 创建 Python 虚拟环境
echo "创建 Python 虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 6. 安装 Python 依赖
echo "安装 Python 依赖..."
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn  # 用于生产环境运行 Flask

# 7. 创建必要的目录
echo "创建必要的目录..."
mkdir -p uploads logs

# 8. 设置权限
echo "设置文件权限..."
chown -R $SERVICE_USER:$SERVICE_USER $DEPLOY_DIR 2>/dev/null || chown -R $USER:$USER $DEPLOY_DIR
chmod +x server.py

# 9. 创建 systemd 服务文件
echo "创建 systemd 服务..."
cat > /etc/systemd/system/quantitative-program.service <<EOF
[Unit]
Description=Quantitative Program Trading Signal Server
After=network.target

[Service]
Type=notify
User=$SERVICE_USER
WorkingDirectory=$DEPLOY_DIR
Environment="PATH=$DEPLOY_DIR/venv/bin"
ExecStart=$DEPLOY_DIR/venv/bin/gunicorn --bind 0.0.0.0:8000 --workers 4 --timeout 300 --access-logfile $DEPLOY_DIR/logs/access.log --error-logfile $DEPLOY_DIR/logs/error.log server:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 10. 重新加载 systemd
echo "重新加载 systemd..."
systemctl daemon-reload

# 11. 启动服务
echo "启动服务..."
systemctl enable quantitative-program
systemctl start quantitative-program

# 12. 检查服务状态
echo "检查服务状态..."
sleep 2
systemctl status quantitative-program --no-pager

echo "=========================================="
echo "部署完成！"
echo "=========================================="
echo "服务状态: systemctl status quantitative-program"
echo "查看日志: journalctl -u quantitative-program -f"
echo "重启服务: systemctl restart quantitative-program"
echo "停止服务: systemctl stop quantitative-program"
echo ""
echo "API 地址: http://YOUR_SERVER_IP:8000"
echo "健康检查: http://YOUR_SERVER_IP:8000/health"
echo "=========================================="

