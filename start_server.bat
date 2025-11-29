@echo off
chcp 65001 >nul
echo ========================================
echo 量化交易信号预测服务器
echo ========================================
echo.
echo 正在检查 Python 环境...
python --version
if errorlevel 1 (
    echo 错误: 未找到 Python，请先安装 Python
    pause
    exit /b 1
)
echo.
echo 正在检查依赖...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 错误: 依赖安装失败
        pause
        exit /b 1
    )
)
echo.
echo 正在启动服务器...
echo.
python server.py
pause

