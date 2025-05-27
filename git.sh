# 进入当前脚本所在的目录
path=$(dirname $0)
echo $path
cd $path

# 添加所有文件
git add .

# 提交并推送到远程仓库
git -c user.name="richardodliu" -c user.email="richardodliu@gmail.com" commit -am "update"
git push origin main