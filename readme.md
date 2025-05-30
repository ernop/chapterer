# 📚 EPUB Chapter Manager

> **Clean, intelligent EPUB text extraction optimized for Large Language Models**

Transform your EPUB books into clean, ASCII text files perfectly sized for Claude, ChatGPT, and other LLMs. Select exactly which sections to include, get real-time token counting, and export with smart filenames.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)

## ✨ Features

### 🎯 **LLM-Optimized**
- **Claude 3/4 ready** - Built-in token limits and safety checks
- **Clean ASCII output** - Converts fancy quotes, em-dashes, and Unicode to standard characters
- **Smart text extraction** - Removes HTML formatting while preserving paragraph structure
- **Token counting** - Real-time estimates for Claude's 200K context window

### 🎛️ **Interactive Selection**
- **Beautiful chapter browser** with previews and token counts
- **Flexible selection** - Include/exclude individual sections or ranges
- **Auto-fit mode** - Automatically select sections to fit Claude limits
- **Command history** - Full readline support with ↑↓ navigation and editing

### 📁 **Smart Export**
- **Descriptive filenames** - `BookTitle-sec1-5.txt`, `Novel-sec3+7+9-12.txt`
- **Preserves book structure** - Uses actual chapter headings, not imposed labels
- **Multiple formats** - Plain text and Markdown support
- **Compatibility warnings** - Know if your export fits in Claude's context window

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install ebooklib beautifulsoup4

# On Windows, you might also need:
pip install pyreadline3
```

### Basic Usage

```bash
# Interactive mode - explore and select chapters
python epub_manager.py book.epub

# Quick export with auto-sizing for Claude
python epub_manager.py book.epub --auto-fit --export

# Export all chapters
python epub_manager.py book.epub --include-all --export complete-book.txt
```

## 📖 Interactive Commands

| Command | Description | Example |
|---------|-------------|---------|
| `show` | Display all sections with previews | `show` |
| `preview <num>` | Detailed view of a section | `preview 5` |
| `toggle <num>` | Toggle section inclusion | `toggle 3` |
| `inc <num\|range>` | Include sections | `inc 1-10` |
| `exc <num\|range>` | Exclude sections | `exc 5` |
| `auto-fit` | Auto-select f