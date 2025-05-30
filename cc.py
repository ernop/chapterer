#!/usr/bin/env python3
"""
EPUB Chapter Manager
Reads EPUB files, allows selective chapter inclusion/exclusion, and exports selected content.
Optimized for Claude 3/4 token limits and LLM usage.
"""

import os
import sys
import re
from typing import List, Dict, Tuple
import argparse
from pathlib import Path

try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    import readline  # For command history and editing
except ImportError:
    missing = []
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError:
        missing.append("ebooklib")
    
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        missing.append("beautifulsoup4")
    
    try:
        import readline
    except ImportError:
        missing.append("readline (usually built-in on Unix systems)")
    
    if missing:
        print("Required libraries not found. Install with:")
        print(f"pip install {' '.join([lib for lib in missing if lib != 'readline (usually built-in on Unix systems)'])}")
        if "readline" in str(missing):
            print("Note: readline should be built-in on Unix/Linux/Mac. On Windows, try: pip install pyreadline3")
        sys.exit(1)


class TokenLimits:
    """Claude model token limits for reference"""
    
    # Context window limits (input)
    CLAUDE_3_CONTEXT = 200_000  # ~150k words, ~500 pages
    CLAUDE_4_CONTEXT = 200_000  # Same as Claude 3
    
    # Output limits
    CLAUDE_3_OUTPUT = 4_096    # Standard
    CLAUDE_3_5_OUTPUT = 8_192  # With beta header
    CLAUDE_4_SONNET_OUTPUT = 64_000  # Up to 64K output tokens
    CLAUDE_4_OPUS_OUTPUT = 32_000    # Up to 32K output tokens
    
    # Rough token-to-character ratio (varies by content)
    CHARS_PER_TOKEN = 4  # Conservative estimate
    
    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """Estimate token count from character count"""
        return len(text) // cls.CHARS_PER_TOKEN
    
    @classmethod
    def get_safe_limits(cls) -> Dict[str, int]:
        """Get safe character limits for different Claude models"""
        return {
            'claude_3_safe': int(cls.CLAUDE_3_CONTEXT * cls.CHARS_PER_TOKEN * 0.8),  # 80% of limit
            'claude_4_safe': int(cls.CLAUDE_4_CONTEXT * cls.CHARS_PER_TOKEN * 0.8),
            'claude_3_chars': cls.CLAUDE_3_CONTEXT * cls.CHARS_PER_TOKEN,
            'claude_4_chars': cls.CLAUDE_4_CONTEXT * cls.CHARS_PER_TOKEN,
        }


class EPUBManager:
    def __init__(self, epub_path: str):
        self.epub_path = epub_path
        self.book = None
        self.chapters = []
        self.chapter_states = {}  # True = include, False = exclude
        self.book_title = ""
        self.book_author = ""
        self.load_epub()
        self.setup_readline()
    
    def setup_readline(self):
        """Configure readline for command history and editing"""
        try:
            # Enable tab completion
            readline.parse_and_bind('tab: complete')
            # Enable history
            readline.parse_and_bind('set editing-mode emacs')
            # Load history if it exists
            history_file = os.path.expanduser('~/.epub_manager_history')
            try:
                readline.read_history_file(history_file)
            except FileNotFoundError:
                pass
            # Set history length
            readline.set_history_length(1000)
        except Exception:
            pass  # readline might not be available on all systems
    
    def save_readline_history(self):
        """Save command history"""
        try:
            history_file = os.path.expanduser('~/.epub_manager_history')
            readline.write_history_file(history_file)
        except Exception:
            pass
    
    def clean_text_content(self, html_content: str) -> str:
        """Extract and clean text content from HTML, optimized for LLM consumption."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Handle common block elements that should create paragraph breaks
        for tag in soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'li']):
            tag.insert_after('\n\n')
        
        # Handle line breaks
        for br in soup.find_all('br'):
            br.replace_with('\n')
        
        # Get text content
        text = soup.get_text()
        
        # Convert fancy characters to ASCII equivalents
        text = self.normalize_to_ascii(text)
        
        # Clean up whitespace and formatting
        # Replace multiple spaces/tabs with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Normalize line breaks - ensure proper paragraph separation
        # First, replace multiple consecutive newlines with double newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Handle single newlines more carefully
        # Keep newlines that are likely intentional paragraph breaks
        lines = text.split('\n')
        processed_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # If this line ends with sentence punctuation and next line
            # doesn't start with lowercase, it's likely a paragraph break
            if (i < len(lines) - 1 and 
                line and 
                line[-1] in '.!?"' and 
                i + 1 < len(lines) and 
                lines[i + 1].strip() and 
                not lines[i + 1].strip()[0].islower()):
                processed_lines.append(line)
                processed_lines.append('')  # Add paragraph break
            else:
                processed_lines.append(line)
        
        # Join and clean up
        text = '\n'.join(processed_lines)
        
        # Final cleanup: ensure proper paragraph separation
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Single newlines become spaces
        
        # Clean up any remaining multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Split into paragraphs and clean each one
        paragraphs = text.split('\n\n')
        clean_paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Ensure we have proper paragraph separation
        result = '\n\n'.join(clean_paragraphs)
        
        # Final safety check - if we have very long lines without breaks,
        # try to add some reasonable breaks at sentence boundaries
        if any(len(p) > 1000 and '. ' in p for p in clean_paragraphs):
            final_paragraphs = []
            for p in clean_paragraphs:
                if len(p) > 1000 and '. ' in p:
                    # Split long paragraphs at sentence boundaries
                    sentences = re.split(r'(?<=[.!?])\s+', p)
                    current_para = ''
                    for sentence in sentences:
                        if len(current_para) + len(sentence) > 800:
                            if current_para:
                                final_paragraphs.append(current_para.strip())
                                current_para = sentence
                            else:
                                final_paragraphs.append(sentence.strip())
                        else:
                            current_para += (' ' if current_para else '') + sentence
                    if current_para:
                        final_paragraphs.append(current_para.strip())
                else:
                    final_paragraphs.append(p)
            result = '\n\n'.join(final_paragraphs)
        
        return result
    
    def normalize_to_ascii(self, text: str) -> str:
        """Convert fancy Unicode characters to ASCII equivalents."""
        # Smart quotes to straight quotes
        text = text.replace('"', '"').replace('"', '"')  # Smart double quotes
        text = text.replace(''', "'").replace(''', "'")  # Smart single quotes
        
        # Em dash and en dash to regular dash
        text = text.replace('—', '--').replace('–', '-')
        
        # Ellipsis
        text = text.replace('…', '...')
        
        # Various spaces to regular space
        text = text.replace('\u00A0', ' ')  # Non-breaking space
        text = text.replace('\u2009', ' ')  # Thin space
        text = text.replace('\u2002', ' ')  # En space
        text = text.replace('\u2003', ' ')  # Em space
        
        # Fancy apostrophes
        text = text.replace('´', "'").replace('`', "'")
        
        # Other common Unicode replacements
        text = text.replace('©', '(c)').replace('®', '(r)')
        text = text.replace('™', '(tm)')
        
        # Fractions
        text = text.replace('½', '1/2').replace('¼', '1/4').replace('¾', '3/4')
        
        # Convert any remaining non-ASCII characters to closest ASCII equivalent
        import unicodedata
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if ord(c) < 128 or c in ' \n\t')
        
        return text
    
    def load_epub(self):
        """Load the EPUB file and extract chapters."""
        try:
            self.book = epub.read_epub(self.epub_path)
            print(f"Loading EPUB: {os.path.basename(self.epub_path)}")
            
            # Get book metadata
            title_meta = self.book.get_metadata('DC', 'title')
            author_meta = self.book.get_metadata('DC', 'creator')
            
            self.book_title = title_meta[0][0] if title_meta else os.path.splitext(os.path.basename(self.epub_path))[0]
            self.book_author = author_meta[0][0] if author_meta else "Unknown Author"
            
            # Clean up title for filename use
            self.book_title = re.sub(r'[^\w\s-]', '', self.book_title).strip()
            self.book_title = re.sub(r'[-\s]+', '-', self.book_title)
            
            # Extract chapters (HTML documents)
            chapter_num = 1
            for item in self.book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # Clean text content for LLM consumption
                    clean_text = self.clean_text_content(item.get_content())
                    
                    # Skip very short chapters (likely front matter, TOC, etc.)
                    if len(clean_text.strip()) < 200:
                        continue
                    
                    # Try to get a meaningful title
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    title = item.get_name()
                    
                    if soup.title and soup.title.string:
                        title = soup.title.string.strip()
                    elif soup.h1:
                        title = soup.h1.get_text(strip=True)[:50]
                    elif soup.h2:
                        title = soup.h2.get_text(strip=True)[:50]
                    elif soup.h3:
                        title = soup.h3.get_text(strip=True)[:50]
                    
                    # Clean title
                    title = re.sub(r'\s+', ' ', title).strip()
                    if not title or title == item.get_name():
                        title = f"Section {chapter_num}"
                    
                    # Get preview text (first 300 chars)
                    preview = self._get_chapter_preview(clean_text)
                    
                    chapter_info = {
                        'number': chapter_num,
                        'title': title,
                        'filename': item.get_name(),
                        'content': clean_text,
                        'length': len(clean_text),
                        'tokens': TokenLimits.estimate_tokens(clean_text),
                        'preview': preview,
                        'item': item
                    }
                    
                    self.chapters.append(chapter_info)
                    self.chapter_states[chapter_num] = True  # Include by default
                    chapter_num += 1
                    
        except Exception as e:
            print(f"Error loading EPUB: {e}")
            sys.exit(1)
    
    def _get_chapter_preview(self, text: str, max_length: int = 150) -> str:
        """Extract a clean preview of the chapter content for display in listings."""
        # Clean up the text
        text = text.strip()
        
        # Try to get a full sentence
        if len(text) <= max_length:
            return text
        
        # Find the last sentence boundary within the limit
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')
        
        boundary = max(last_period, last_exclamation, last_question)
        
        if boundary > max_length * 0.6:  # If we found a sentence boundary in the latter part
            return text[:boundary + 1]
        else:
            # Try to end at a word boundary
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:
                return text[:last_space] + "..."
            return truncated + "..."
    
    def show_file_info(self):
        """Display basic information about the EPUB file and Claude limits."""
        file_size = os.path.getsize(self.epub_path)
        total_chars = sum(chapter['length'] for chapter in self.chapters)
        total_tokens = sum(chapter['tokens'] for chapter in self.chapters)
        limits = TokenLimits.get_safe_limits()
        
        print(f"\n📖 EPUB Information:")
        print(f"   File: {os.path.basename(self.epub_path)}")
        print(f"   Title: {self.book_title}")
        print(f"   Author: {self.book_author}")
        print(f"   Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        print(f"   Sections: {len(self.chapters)} (after filtering)")
        print(f"   Total clean text: {total_chars:,} characters (~{total_tokens:,} tokens)")
        
        # Show Claude compatibility
        print(f"\n🤖 Claude Model Compatibility:")
        print(f"   Context Window: 200,000 tokens (~{limits['claude_3_chars']:,} chars)")
        print(f"   Safe Limit (80%): ~{limits['claude_3_safe']:,} chars")
        
        if total_chars > limits['claude_3_safe']:
            print(f"   ⚠️  Full book exceeds safe limit by {total_chars - limits['claude_3_safe']:,} chars")
            print(f"   📝 Use chapter selection to stay within limits")
        else:
            print(f"   ✅ Full book fits within safe limits")
    
    def show_chapters(self):
        """Display all chapters with their inclusion state, length, and preview."""
        included_chars = sum(ch['length'] for ch in self.chapters if self.chapter_states[ch['number']])
        included_tokens = sum(ch['tokens'] for ch in self.chapters if self.chapter_states[ch['number']])
        included_count = sum(1 for ch in self.chapters if self.chapter_states[ch['number']])
        limits = TokenLimits.get_safe_limits()
        
        print(f"\n📑 Sections ({len(self.chapters)} total, {included_count} selected):")
        print(f"   Selected: {included_chars:,} chars (~{included_tokens:,} tokens)")
        
        if included_chars > limits['claude_3_safe']:
            print(f"   ⚠️  Selection exceeds safe limit by {included_chars - limits['claude_3_safe']:,} chars")
        else:
            print(f"   ✅ Selection within safe limits")
        
        print(f"\n   #  | Status | Length  | Tokens | Title & Preview")
        print(f"   ---|--------|---------|--------|{'-' * 70}")
        
        for chapter in self.chapters:
            num = chapter['number']
            status = "[+] INCL" if self.chapter_states[num] else "[-] EXCL"
            length = f"{chapter['length']:,}".rjust(7)
            tokens = f"{chapter['tokens']:,}".rjust(6)
            title = chapter['title'][:35] + "..." if len(chapter['title']) > 38 else chapter['title']
            
            # Format preview on same line
            preview = chapter['preview']
            if len(preview) > 75:
                preview = preview[:72] + "..."
            
            print(f"   {num:2d} | {status} | {length} | {tokens} | {title} -> {preview}")
    
    def show_chapter_preview(self, chapter_num: int):
        """Show detailed preview of a specific section."""
        if not (1 <= chapter_num <= len(self.chapters)):
            print(f"Invalid section number: {chapter_num}")
            return
        
        chapter = self.chapters[chapter_num - 1]
        status = "INCLUDED" if self.chapter_states[chapter_num] else "EXCLUDED"
        
        print(f"\n📄 Section {chapter_num} Preview:")
        print(f"   Title: {chapter['title']}")
        print(f"   Length: {chapter['length']:,} characters (~{chapter['tokens']:,} tokens)")
        print(f"   Status: {status}")
        print(f"   Filename: {chapter['filename']}")
        print(f"\n   Clean Text Preview:")
        print(f"   {'-' * 70}")
        print(f"   {chapter['preview']}")
        print(f"   {'-' * 70}")
    
    def toggle_chapter(self, chapter_num: int) -> bool:
        """Toggle inclusion state of a section. Returns True if successful."""
        if 1 <= chapter_num <= len(self.chapters):
            self.chapter_states[chapter_num] = not self.chapter_states[chapter_num]
            return True
        return False
    
    def toggle_range(self, start: int, end: int, include: bool):
        """Set inclusion state for a range of sections."""
        for num in range(start, end + 1):
            if 1 <= num <= len(self.chapters):
                self.chapter_states[num] = include
    
    def generate_smart_filename(self, base_filename: str = None) -> str:
        """Generate a descriptive filename based on selected sections."""
        if base_filename:
            base_name = os.path.splitext(base_filename)[0]
            extension = os.path.splitext(base_filename)[1] or '.txt'
        else:
            base_name = None
            extension = '.txt'
        
        # Get included section ranges
        included_chapters = sorted([ch['number'] for ch in self.chapters if self.chapter_states[ch['number']]])
        
        if not included_chapters:
            range_desc = "empty"
        elif len(included_chapters) == len(self.chapters):
            range_desc = "complete"
        else:
            # Find consecutive ranges
            ranges = []
            start = included_chapters[0]
            prev = start
            
            for num in included_chapters[1:] + [None]:  # Add None to trigger final range
                if num is None or num != prev + 1:
                    if start == prev:
                        ranges.append(str(start))
                    else:
                        ranges.append(f"{start}-{prev}")
                    if num is not None:
                        start = num
                prev = num
            
            range_desc = "sec" + "+".join(ranges)
        
        # Create filename
        book_name = self.book_title[:30]  # Limit length
        if base_name:
            filename = f"{base_name}-{range_desc}{extension}"
        else:
            filename = f"{book_name}-{range_desc}{extension}"
        
        # Clean filename
        filename = re.sub(r'[^\w\s.-]', '', filename)
        filename = re.sub(r'[-\s]+', '-', filename)
        
        return filename
    
    def export_selected_chapters(self, output_path: str = None, format_type: str = 'txt'):
        """Export selected chapters to a clean text file optimized for LLM input."""
        included_chapters = [ch for ch in self.chapters if self.chapter_states[ch['number']]]
        
        if not included_chapters:
            print("⚠️  No chapters selected for export!")
            return False
        
        # Generate smart filename if not provided
        if not output_path:
            output_path = self.generate_smart_filename()
        else:
            output_path = self.generate_smart_filename(output_path)
        
        # Check if export will fit in Claude limits
        total_chars = sum(ch['length'] for ch in included_chapters)
        total_tokens = sum(ch['tokens'] for ch in included_chapters)
        limits = TokenLimits.get_safe_limits()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Minimal header for LLM consumption
                f.write(f"Title: {self.book_title}\n")
                f.write(f"Author: {self.book_author}\n")
                f.write(f"Chapters: {len(included_chapters)} of {len(self.chapters)}\n")
                f.write(f"Content: {total_chars:,} characters (~{total_tokens:,} tokens)\n")
                
                # Claude compatibility note
                if total_chars <= limits['claude_3_safe']:
                    f.write("Status: Claude-optimized (within safe limits)\n")
                elif total_chars <= limits['claude_3_chars']:
                    f.write("Status: At Claude context limit\n")
                else:
                    f.write("Status: Exceeds Claude context window\n")
                
                f.write("\n" + "=" * 60 + "\n\n")
                
                # Export clean chapter content
                for i, chapter in enumerate(included_chapters):
                    # Simple chapter header for LLM parsing
                    f.write(f"CHAPTER {chapter['number']}: {chapter['title']}\n\n")
                    
                    # Clean content
                    f.write(chapter['content'])
                    
                    # Chapter separator (except for last chapter)
                    if i < len(included_chapters) - 1:
                        f.write(f"\n\n{'='*60}\n\n")
            
            print(f"✅ Exported to: {output_path}")
            print(f"   Content: {len(included_chapters)} chapters, {total_chars:,} chars (~{total_tokens:,} tokens)")
            
            # Show Claude compatibility
            if total_chars > limits['claude_3_safe']:
                print(f"   ⚠️  Warning: Exceeds safe Claude limit by {total_chars - limits['claude_3_safe']:,} chars")
                print(f"   💡 Consider reducing selection for optimal LLM performance")
            else:
                print(f"   ✅ Optimized for Claude LLM input")
            
            return True
            
        except Exception as e:
            print(f"❌ Error exporting: {e}")
            return False
    
    def auto_fit_selection(self):
        """Automatically select sections to fit within Claude safe limits."""
        limits = TokenLimits.get_safe_limits()
        safe_limit = limits['claude_3_safe']
        
        # Reset all to excluded
        for num in self.chapter_states:
            self.chapter_states[num] = False
        
        current_chars = 0
        included_count = 0
        
        # Include sections in order until we approach the limit
        for chapter in self.chapters:
            if current_chars + chapter['length'] <= safe_limit:
                self.chapter_states[chapter['number']] = True
                current_chars += chapter['length']
                included_count += 1
            else:
                break
        
        print(f"🎯 Auto-selected {included_count} sections ({current_chars:,} chars) to fit Claude limits")
        return included_count
    
    def run_interactive_mode(self):
        """Run the interactive command-line interface with readline support."""
        print("🎛️  Interactive Chapter Manager (Use ↑↓ for history, ←→ to edit)")
        print("Commands: show, preview <num>, toggle <num>, inc/exc <num|range>, auto-fit, export [filename], quit")
        
        while True:
            try:
                cmd = input("\n📚 > ").strip()
                
                if not cmd:
                    continue
                
                cmd_lower = cmd.lower()
                
                if cmd_lower in ['q', 'quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                elif cmd_lower in ['s', 'show']:
                    self.show_chapters()
                
                elif cmd_lower.startswith('p ') or cmd_lower.startswith('preview '):
                    try:
                        num = int(cmd.split()[1])
                        self.show_chapter_preview(num)
                    except (IndexError, ValueError):
                        print("Usage: preview <chapter_number>")
                
                elif cmd_lower.startswith('t ') or cmd_lower.startswith('toggle '):
                    try:
                        num = int(cmd.split()[1])
                        if self.toggle_chapter(num):
                            state = "included" if self.chapter_states[num] else "excluded"
                            print(f"Section {num} is now {state}")
                        else:
                            print(f"Invalid section number: {num}")
                    except (IndexError, ValueError):
                        print("Usage: toggle <section_number>")
                
                elif cmd_lower.startswith('include ') or cmd_lower.startswith('exclude ') or cmd_lower.startswith('inc ') or cmd_lower.startswith('exc '):
                    try:
                        parts = cmd.split()
                        range_part = parts[1]
                        include = cmd_lower.startswith('include') or cmd_lower.startswith('inc')
                        
                        if '-' in range_part:
                            start, end = map(int, range_part.split('-'))
                            self.toggle_range(start, end, include)
                            action = "included" if include else "excluded"
                            print(f"Sections {start}-{end} are now {action}")
                        else:
                            num = int(range_part)
                            if 1 <= num <= len(self.chapters):
                                self.chapter_states[num] = include
                                action = "included" if include else "excluded"
                                print(f"Section {num} is now {action}")
                            else:
                                print(f"Invalid section number: {num}")
                    except (IndexError, ValueError):
                        print("Usage: include/exclude/inc/exc <num> or <start>-<end>")
                
                elif cmd_lower in ['auto', 'auto-fit', 'autofit']:
                    self.auto_fit_selection()
                
                elif cmd_lower.startswith('e ') or cmd_lower.startswith('export ') or cmd_lower == 'export':
                    try:
                        parts = cmd.split(None, 1)
                        filename = parts[1] if len(parts) > 1 else None
                        self.export_selected_chapters(filename)
                    except Exception as e:
                        print(f"Export error: {e}")
                
                elif cmd_lower in ['limits', 'limit', 'l']:
                    limits = TokenLimits.get_safe_limits()
                    included_chars = sum(ch['length'] for ch in self.chapters if self.chapter_states[ch['number']])
                    print(f"\n🤖 Claude Token Limits:")
                    print(f"   Context Window: 200,000 tokens (~{limits['claude_3_chars']:,} chars)")
                    print(f"   Safe Limit (80%): ~{limits['claude_3_safe']:,} chars")
                    print(f"   Current Selection: {included_chars:,} chars")
                    if included_chars <= limits['claude_3_safe']:
                        print(f"   Remaining Capacity: {limits['claude_3_safe'] - included_chars:,} chars")
                    else:
                        print(f"   Exceeds Safe Limit: {included_chars - limits['claude_3_safe']:,} chars")
                
                elif cmd_lower in ['h', 'help']:
                    print("\nCommands:")
                    print("  show                    - Show all sections with status & previews")
                    print("  preview <num>           - Show detailed preview of section")
                    print("  toggle <num>            - Toggle section inclusion")
                    print("  include <num|start-end> - Include section(s) (short: inc)")
                    print("  exclude <num|start-end> - Exclude section(s) (short: exc)")
                    print("  auto-fit                - Auto-select sections for Claude limits")
                    print("  limits                  - Show Claude token limit info")
                    print("  export [filename]       - Export with smart filename")
                    print("  quit                    - Exit program")
                    print("\nKeyboard shortcuts:")
                    print("  ↑/↓                     - Browse command history")
                    print("  ←/→                     - Move cursor in command")
                    print("  Ctrl+A/E               - Beginning/end of line")
                    print("  Ctrl+K                 - Delete to end of line")
                
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        # Save command history when exiting
        self.save_readline_history()


def main():
    parser = argparse.ArgumentParser(
        description="EPUB Chapter Manager - Extract clean text optimized for Claude LLMs",
        epilog="""
Claude Token Limits:
  • Context Window: 200,000 tokens (~800,000 characters)
  • Safe Limit: 160,000 tokens (~640,000 characters) 
  • Output Limits: 4K-64K tokens depending on model

Smart Filenames:
  Book sections 1-5: "BookName-sec1-5.txt"
  Sections 3,7,9-12: "BookName-sec3+7+9-12.txt"
  All sections: "BookName-complete.txt"

Examples:
  python epub_manager.py book.epub
  python epub_manager.py book.epub --auto-fit --export
  python epub_manager.py book.epub --exclude-all
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("epub_file", help="Path to the EPUB file")
    parser.add_argument("--export", "-e", nargs='?', const='', help="Export selected chapters (optional filename)")
    parser.add_argument("--include-all", action="store_true", help="Include all chapters by default")
    parser.add_argument("--exclude-all", action="store_true", help="Exclude all chapters by default")
    parser.add_argument("--auto-fit", action="store_true", help="Auto-select chapters to fit Claude limits")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.epub_file):
        print(f"Error: File not found: {args.epub_file}")
        sys.exit(1)
    
    # Initialize manager
    manager = EPUBManager(args.epub_file)
    
    # Set initial states if specified
    if args.exclude_all:
        for num in manager.chapter_states:
            manager.chapter_states[num] = False
    elif args.include_all:
        for num in manager.chapter_states:
            manager.chapter_states[num] = True
    elif args.auto_fit:
        manager.auto_fit_selection()
    
    # Show file info
    manager.show_file_info()
    
    # If export specified, do quick export
    if args.export is not None:
        manager.show_chapters()
        filename = args.export if args.export else None
        manager.export_selected_chapters(filename)
    else:
        # Run interactive mode
        manager.show_chapters()
        manager.run_interactive_mode()


if __name__ == "__main__":
    main()