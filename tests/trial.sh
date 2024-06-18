git branch trial
git switch trial
git push --set-upstream origin trial
touch text.txt
git add ./text.txt
git commit -m "testing file"
rm text.txt
git add .
git push origin
git switch main
# git git push origin --delete trial  