# MLOps ICU - DVC Initialization Script

# 1. Initialize Git (required for DVC)
git init

# 2. Initialize DVC
dvc init

# 3. Set up the local remote storage
# Using the requested path: C:\Users\21270\Desktop\mlops\dvc-storage
$remotePath = "C:\Users\21270\Desktop\mlops\dvc-storage"
if (!(Test-Path $remotePath)) { New-Item -ItemType Directory -Path $remotePath }

dvc remote add -d localremote $remotePath

# 4. Add data to DVC
dvc add data/

# 5. Track changes in Git
git add data.dvc .dvc/config .gitignore
git commit -m "Initialize DVC with raw and processed data"

# 6. Push to local remote
dvc push
