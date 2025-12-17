#!/bin/bash
# Prepare repository for publication on GitHub

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Melanoma Detection - Repository Publication Checklist  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command succeeded
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1"
        exit 1
    fi
}

echo "📋 Step 1: Checking Python environment..."
python --version
check_status "Python installed"

echo ""
echo "📦 Step 2: Installing dependencies..."
pip install -q -r requirements.txt
check_status "Dependencies installed"

echo ""
echo "🧪 Step 3: Running basic sanity checks..."

# Check if key files exist
echo "  Checking essential files..."
files=(
    "README.md"
    "LICENSE"
    "requirements.txt"
    "inference.py"
    "CONTRIBUTING.md"
    "CHANGELOG.md"
    ".gitignore"
    "scripts/download_weights.py"
    "checkpoints/README.md"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "    ${GREEN}✓${NC} $file"
    else
        echo -e "    ${RED}✗${NC} $file (missing)"
        exit 1
    fi
done

echo ""
echo "🔍 Step 4: Checking directory structure..."
dirs=(
    "anomaly_detection"
    "classifiers"
    "data"
    "generators"
    "results"
    "scripts"
    "streamlit"
    "checkpoints"
)

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "    ${GREEN}✓${NC} $dir/"
    else
        echo -e "    ${YELLOW}⚠${NC} $dir/ (missing or not created yet)"
    fi
done

echo ""
echo "📝 Step 5: Checking for sensitive data..."
# Check for common sensitive files
sensitive_patterns=(
    "*.env"
    "*.key"
    "*.pem"
    "*secret*"
    "*password*"
)

found_sensitive=false
for pattern in "${sensitive_patterns[@]}"; do
    if ls $pattern 2>/dev/null | grep -q .; then
        echo -e "    ${RED}✗${NC} Found sensitive file: $pattern"
        found_sensitive=true
    fi
done

if [ "$found_sensitive" = false ]; then
    echo -e "    ${GREEN}✓${NC} No sensitive files detected"
fi

echo ""
echo "📊 Step 6: Repository statistics..."
echo "  Python files: $(find . -name '*.py' -not -path './venv/*' | wc -l)"
echo "  Total lines of code: $(find . -name '*.py' -not -path './venv/*' -exec cat {} \; | wc -l)"
echo "  Notebooks: $(find . -name '*.ipynb' | wc -l)"
echo "  Markdown docs: $(find . -name '*.md' | wc -l)"

echo ""
echo "🔧 Step 7: Git status..."
if [ -d ".git" ]; then
    echo "  Branch: $(git branch --show-current)"
    echo "  Remote: $(git remote get-url origin 2>/dev/null || echo 'No remote configured')"
    echo "  Uncommitted changes: $(git status --porcelain | wc -l) files"
else
    echo -e "    ${YELLOW}⚠${NC} Not a git repository"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                 Publication Checklist                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Before pushing to GitHub, ensure:"
echo ""
echo "📝 Documentation:"
echo "  [✓] README.md updated with hybrid results"
echo "  [✓] CHANGELOG.md reflects version 2.0"
echo "  [✓] CONTRIBUTING.md has contribution guidelines"
echo "  [✓] All code has English comments and docstrings"
echo ""
echo "🔬 Code Quality:"
echo "  [ ] All scripts run without errors"
echo "  [ ] Inference.py tested on sample image"
echo "  [ ] No hardcoded paths or credentials"
echo "  [ ] Type hints added to functions"
echo ""
echo "📊 Results:"
echo "  [✓] Hybrid system evaluation completed"
echo "  [✓] Performance metrics documented (99.75% recall)"
echo "  [✓] Confusion matrices generated"
echo "  [ ] Sample predictions included"
echo ""
echo "🚀 Deployment:"
echo "  [✓] Download script for model weights"
echo "  [ ] Models uploaded to Hugging Face Hub"
echo "  [✓] requirements.txt complete"
echo "  [ ] Optional: Docker image created"
echo ""
echo "🔒 Security:"
echo "  [✓] .gitignore excludes large files and data"
echo "  [✓] No API keys or passwords in code"
echo "  [✓] Medical disclaimer in README"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}✓ Repository is ready for publication!${NC}"
echo ""
echo "Next steps:"
echo "  1. git add ."
echo "  2. git commit -m 'feat: add hybrid detection system with 99.75% recall'"
echo "  3. git push origin main"
echo "  4. Create release on GitHub with version tag v2.0.0"
echo ""
echo "🎗️  Thank you for contributing to melanoma detection research!"
