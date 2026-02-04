先执行 git clone https://github.com/zczjyq/DeepASM-Net.git
cd DeepASM-Net

如果是用zip的
git init
git remote add origin https://github.com/zczjyq/DeepASM-Net.git

然后
git reset --mixed origin/main
git status 看一下没东西说明成功了
# 建议使用和你 GitHub 账号一致的邮箱和用户名
git config --global user.email "你的邮箱@example.com"
git config --global user.name "你的名字"
git push --set-upstream origin main


注意配置.gitignore
.gitignore 在仓库里面有


下载更新 git pull
暂存改动 git add .
本地保存 git commit -m ""
上传云端 git push

推送前先pull！！!!

防止冲突：git add . → git commit -m "" → git pull --rebase → 解决冲突(如果有) → git push

git pull --rebase含义：如果他已经 push 了，就先把他的改动接过来，再把我的提交接在最上面。
