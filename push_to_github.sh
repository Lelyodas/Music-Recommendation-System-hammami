#!/bin/bash

# Script to push the committed changes to GitHub
# Run this script after GitHub credentials are configured

echo "=========================================="
echo "Pushing to GitHub Repository"
echo "=========================================="
echo ""
echo "Repository: Music-Recommendation-System-hammami"
echo "Branch: main"
echo "Commit: 1f94dd5 - Complete music data analysis project"
echo ""

cd /home/user/webapp

# Check if there are commits to push
if git log origin/main..HEAD 2>/dev/null | grep -q .; then
    echo "Pushing commits to remote repository..."
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ SUCCESS! Changes pushed to GitHub."
        echo ""
        echo "View your repository at:"
        echo "https://github.com/Lelyodas/Music-Recommendation-System-hammami"
    else
        echo ""
        echo "❌ Push failed. Please check your GitHub credentials."
        echo ""
        echo "To configure credentials, you may need to:"
        echo "1. Generate a GitHub Personal Access Token"
        echo "2. Use 'git credential' to store it"
        echo "3. Or set up SSH keys for authentication"
    fi
else
    echo "Attempting to push..."
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ SUCCESS! Changes pushed to GitHub."
        echo ""
        echo "View your repository at:"
        echo "https://github.com/Lelyodas/Music-Recommendation-System-hammami"
    else
        echo ""
        echo "❌ Push failed. Please check your GitHub credentials."
    fi
fi

echo ""
echo "=========================================="
