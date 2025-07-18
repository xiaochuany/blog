---
date:
    created: 2025-04-05
authors: [xy]
categories:  [TIL]
tags: [dev tools]
---

# Fork, merge and PR

<!-- more -->
This is a common workflow: 
- fork a repo (`gh repo fork` with the gh cli fork the repo interactively)
- work on some code then `git add`, `git commit`
- fetch upstream (`git remote add upstream REPO` then `git fetch upstream`)
- merge upstream (`git merge` say `upstream/main`) 
- push back to the fork (`git push`)
- ask for a pull request (`gh pr create` interactively) 

Unfortunately the merge step often result in conflicts, and manual intervention is necessary. Fortunately, editors like vscode offer GUI to help visualize the process, making this step less of a pain (after messing up once/twice to understand the GUI).  

Alternative workflow (after fetch upstream)

- `git rebase upstream/main`  (then resolve conflicts, if any, commit by commit)
- `git push --force-with-lease`
- PR

To avoid fetch and rebase everytime, use 

```sh
git config --global pull.rebase true && git pull upstream main
```