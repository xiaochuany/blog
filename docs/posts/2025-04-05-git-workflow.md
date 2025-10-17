---
date:
    created: 2025-04-05
authors: [xy]
categories:  [Tutorial]
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

- ... same until fetch
- rebase instead of merge: `git rebase upstream/main`  (then resolve conflicts, if any, commit by commit)
- `git push --force-with-lease`
- ... PR as before

To avoid fetch and rebase everytime, use 

```sh
git config --global pull.rebase true && git pull upstream main
```

## How to undo changes

Conceptually, we may think of git as managing three copies of the same project, they are called 

- working directory  
- index/staging area
- HEAD

Initially they are all the same thing. After some edits, the working directory changes but the index and HEAD remain untouched.

Once we run `git add`, the edits are synced to index so working directory and the index are the same thing. 

Once we run `git commit`, the edits are synced to the HEAD and we are once again in a state of all being equal. 

At any point in time of this process, we can undo changes. 

- `git restore .` removes modifications in the current directory that are not yet staged. 
- Once staged (after `git add`), we can drop these changes using `git restore --staged .`  
- Once commit, we have all three copies the same thing, but we can still drop the changes by restore the previous version of the HEAD (or any other version before) `git reset --soft HEAD~1` 


## What happens during the fetch and merge process

Again conceptually, we may think of remote branch yet another copy of the same project. When we run `git fetch`, we sync the remote branch with the latest content
in the remote repository. 
Assuming we fetch remote content before edits, the working directory is same as index and HEAD, `git merge` would compare remote with local and merge them. It's similar if you decide to rebase. 
