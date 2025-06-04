# change to the directory of the script
path=$(dirname $0)
echo $path
cd $path

# add all files
git add .

# read name and email
name="richardodliu"
email="richardodliu@gmail.com"
message="update"

# commit and push to remote repository
git -c user.name="$name" -c user.email="$email" commit -am "$message"
git push origin main