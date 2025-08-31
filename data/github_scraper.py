import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
from typing import List, Dict
from pathlib import Path

class GitHubDocsScraper:
    def __init__(self):
        self.base_url = "https://docs.github.com"
        self.session = None
        self.scraped_urls = set()
        self.documents = []
        
        # Target sections to scrape
        self.target_sections = {
            "getting-started": {
                "url": "/en/get-started",
                "title": "Getting Started",
                "max_pages": 20  # Limit pages per section
            },
            "authentication": {
                "url": "/en/authentication",
                "title": "Authentication & Security",
                "max_pages": 15
            },
            "repositories": {
                "url": "/en/repositories",
                "title": "Repository Management", 
                "max_pages": 25
            },
            "pull-requests": {
                "url": "/en/pull-requests",
                "title": "Pull Requests",
                "max_pages": 15
            }
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_page(self, url: str) -> str:

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    print(f"Failed to fetch {url}: Status {response.status}")
                    return None
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_content(self, html: str, url: str) -> Dict:

        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title_elem = soup.find('h1') or soup.find('title')
        title = title_elem.get_text().strip() if title_elem else "Unknown"
        
        # Remove navigation, sidebar, and footer elements
        for elem in soup.find_all(['nav', 'aside', 'footer', 'header']):
            elem.decompose()
        
        # Find main content area
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find('div', class_='markdown-body') or
            soup.find('div', {'role': 'main'})
        )
        
        if not main_content:
            main_content = soup
        
        # Extract text content, preserving structure
        content_parts = []
        
        for elem in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'code', 'pre']):
            text = elem.get_text().strip()
            if text and len(text) > 10:  # Filter out very short content
                # Add context for headers
                if elem.name in ['h1', 'h2', 'h3', 'h4']:
                    content_parts.append(f"\n## {text}\n")
                else:
                    content_parts.append(text)
        
        content = ' '.join(content_parts).strip()
        
        # Clean up extra whitespace
        content = ' '.join(content.split())
        
        return {
            'title': title,
            'content': content,
            'url': url,
            'section': self.get_section_from_url(url),
            'word_count': len(content.split())
        }
    
    def get_section_from_url(self, url: str) -> str:

        for section_key, section_data in self.target_sections.items():
            if section_data['url'] in url:
                return section_data['title']
        return "General"
    
    async def discover_section_urls(self, section_url: str, max_pages: int) -> List[str]:

        full_url = urljoin(self.base_url, section_url)
        html = await self.fetch_page(full_url)
        
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        urls = set()
        
        # Find all article links
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href and href.startswith('/en/'):
                # Only include links that are in the same section
                if section_url.replace('/en/', '') in href:
                    full_link_url = urljoin(self.base_url, href)
                    urls.add(full_link_url)
        
        # Limit the number of URLs
        return list(urls)[:max_pages]
    
    async def scrape_section(self, section_key: str, section_data: Dict):
        print(f"\nDiscovering URLs in {section_data['title']}...")
        
        urls = await self.discover_section_urls(
            section_data['url'], 
            section_data['max_pages']
        )
        
        print(f"Found {len(urls)} pages to scrape in {section_data['title']}")
        
        for i, url in enumerate(urls, 1):
            if url in self.scraped_urls:
                continue
                
            print(f"Scraping {i}/{len(urls)}: {url.split('/')[-1]}")
            
            html = await self.fetch_page(url)
            if html:
                doc = self.extract_content(html, url)
                if doc['content'] and len(doc['content']) > 200:  # Only keep substantial content
                    self.documents.append(doc)
                    self.scraped_urls.add(url)
            
            # Rate limiting
            await asyncio.sleep(0.5)
    
    async def scrape_all_sections(self):

        print("Starting GitHub Documentation Scraping...")
        
        for section_key, section_data in self.target_sections.items():
            await self.scrape_section(section_key, section_data)
        
        print(f"\nScraping complete! Collected {len(self.documents)} documents")
        
        # Print summary by section
        section_counts = {}
        for doc in self.documents:
            section = doc['section']
            section_counts[section] = section_counts.get(section, 0) + 1
        
        print("\nDocuments per section:")
        for section, count in section_counts.items():
            print(f" - {section}: {count} documents")
    
    def save_documents(self, filename: str = "github_docs.json"):

        output_path = Path(filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
        
        print(f"\nDocuments saved to {output_path}")
        print(f"Total file size: {output_path.stat().st_size / 1024:.1f} KB")
    
    def create_embedding_ready_data(self, output_file: str = "github_docs_for_embedding.json"):

        embedding_data = []
        
        for doc in self.documents:
            # Split long documents into chunks for better embedding
            content = doc['content']
            words = content.split()
            
            if len(words) <= 300:  # Keep short docs as-is
                embedding_data.append({
                    'content': content,
                    'source': f"GitHub Docs - {doc['title']}",
                    'category': doc['section'].lower().replace(' ', '_'),
                    'url': doc['url']
                })
            else:  # Split long documents
                chunk_size = 250
                for i in range(0, len(words), chunk_size):
                    chunk = ' '.join(words[i:i + chunk_size])
                    if len(chunk.split()) > 50:  # Only keep substantial chunks
                        embedding_data.append({
                            'content': chunk,
                            'source': f"GitHub Docs - {doc['title']} (Part {i//chunk_size + 1})",
                            'category': doc['section'].lower().replace(' ', '_'),
                            'url': doc['url']
                        })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(embedding_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎯 Embedding-ready data saved to {output_file}")
        print(f"📄 {len(embedding_data)} chunks ready for embedding")
        
        return embedding_data

async def main():

    async with GitHubDocsScraper() as scraper:
        await scraper.scrape_all_sections()
        scraper.save_documents()
        embedding_data = scraper.create_embedding_ready_data()
    
    print("github_docs.json - Raw scraped documents")
    print("github_docs_for_embedding.json - Formatted for your RAG system")

if __name__ == "__main__":
    asyncio.run(main())