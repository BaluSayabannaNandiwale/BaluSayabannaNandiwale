# 🚀 How to Upload Your Moodify Project to GitHub

This guide will walk you through the process of uploading your Moodify emotion detection project to GitHub.

## Prerequisites

1. A GitHub account (https://github.com/join)
2. Git installed on your computer
3. Your Moodify project files ready

## Step 1: Create a New Repository on GitHub

1. Go to https://github.com and sign in to your account
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Repository name: `emotion-detection-project`
5. Description: "A comprehensive web application that detects user emotions from face images, speech recordings, and text input"
6. Set to Public (or Private if you prefer)
7. **Important**: Leave "Initialize this repository with a README" **unchecked**
8. Click "Create repository"

## Step 2: Get the Repository URL

After creating the repository, you'll see a page with quick setup instructions. Copy the HTTPS URL which will look like:
```
https://github.com/YOUR_USERNAME/emotion-detection-project.git
```

## Step 3: Connect Your Local Project to GitHub

Open your terminal/command prompt and navigate to your project folder:

```bash
cd "c:\Users\nandi\emotion project"
```

Connect your local repository to the GitHub repository:

```bash
git remote add origin https://github.com/YOUR_USERNAME/emotion-detection-project.git
```

## Step 4: Push Your Code to GitHub

Push your code to GitHub:

```bash
git branch -M main
git push -u origin main
```

## Step 5: Verify the Upload

1. Go to your GitHub repository page
2. Refresh the page
3. You should now see all your project files

## Troubleshooting

### If you get authentication errors:

GitHub now requires personal access tokens instead of passwords. To create one:

1. Go to GitHub Settings
2. Developer Settings
3. Personal Access Tokens
4. Generate New Token
5. Give it repo permissions
6. Copy the token and use it instead of your password when prompted

### If you get "Updates were rejected" error:

This usually happens if there are files on GitHub that don't exist locally. Run:

```bash
git pull origin main --allow-unrelated-histories
```

Then try pushing again.

## Making Updates

To update your repository in the future:

1. Add changes: `git add .`
2. Commit changes: `git commit -m "Description of changes"`
3. Push changes: `git push origin main`

## Congratulations!

Your Moodify project is now live on GitHub and ready to share with the world!