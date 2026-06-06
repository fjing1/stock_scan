#!/bin/zsh
cd "$(dirname "$0")"
/usr/local/bin/python3 scan_stocks.py
echo "----------------------------"
echo "运行完成，按回车关闭窗口..."
read
