"""
Simple DeviantArt scraper.
"""
import asyncio
import aiohttp
from pathlib import Path
import logging

from src.config import Config

logger = logging.getLogger(__name__)


class DeviantArtScraper:
    """Simple DeviantArt image scraper."""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = None
        self.access_token = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self._authenticate()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _authenticate(self):
        """Get access token from DeviantArt."""
        auth_url = "https://www.deviantart.com/oauth2/token"
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': self.config.deviant_client_id,
            'client_secret': self.config.deviant_client_secret
        }
        
        async with self.session.post(auth_url, data=auth_data) as response:
            if response.status == 200:
                data = await response.json()
                self.access_token = data['access_token']
                logger.info("Authenticated with DeviantArt")
            else:
                raise Exception(f"Auth failed: {response.status}")
    
    def _extract_username(self, url: str) -> str:
        """Extract username from DeviantArt URL."""
        # https://www.deviantart.com/lejinvr/favourites
        parts = url.strip('/').split('/')
        for i, part in enumerate(parts):
            if part == 'www.deviantart.com' and i + 1 < len(parts):
                return parts[i + 1]
        raise ValueError(f"Can't extract username from: {url}")
    
    async def scrape_favorites(self, output_dir: str) -> int:
        """Scrape favorite images."""
        if not self.config.deviantart_url:
            logger.error("No DeviantArt URL configured")
            return 0
        
        username = self._extract_username(self.config.deviantart_url)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        downloaded = 0
        offset = 0
        
        while downloaded < self.config.max_images:
            # Get favorites batch
            url = "https://www.deviantart.com/api/v1/oauth2/collections/faves"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            params = {
                'username': username,
                'offset': offset,
                'limit': min(50, self.config.max_images - downloaded)
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    logger.error(f"API error: {response.status}")
                    break
                
                data = await response.json()
                results = data.get('results', [])
                
                if not results:
                    break
                
                # Download images
                for item in results:
                    if downloaded >= self.config.max_images:
                        break
                    
                    # Get image URL
                    content = item.get('content', {})
                    image_url = content.get('src')
                    
                    if not image_url:
                        continue
                    
                    # Download
                    title = item.get('title', 'untitled')
                    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))[:50]
                    filename = f"{downloaded:04d}_{safe_title}.jpg"
                    filepath = output_path / filename
                    
                    try:
                        async with self.session.get(image_url) as img_response:
                            if img_response.status == 200:
                                content = await img_response.read()
                                with open(filepath, 'wb') as f:
                                    f.write(content)
                                downloaded += 1
                                logger.info(f"Downloaded: {filename}")
                    except Exception as e:
                        logger.error(f"Failed to download {image_url}: {e}")
                
                offset += len(results)
                await asyncio.sleep(1)  # Rate limiting
        
        logger.info(f"Downloaded {downloaded} images")
        return downloaded


async def scrape_deviantart(config: Config) -> int:
    """Convenience function to scrape DeviantArt."""
    async with DeviantArtScraper(config) as scraper:
        return await scraper.scrape_favorites(config.data_dir)