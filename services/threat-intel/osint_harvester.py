"""
Automated OSINT (Open Source Intelligence) Harvesting
Collects threat intelligence from public sources
"""

import requests
import tweepy
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import re
import json
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class OSINTHarvester:
    """
    Automated OSINT collection from multiple sources
    - Twitter/X for real-time IOCs
    - GitHub for leaked credentials
    - Pastebin for dumps
    - Shodan for exposed services
    - VirusTotal for malware IOCs
    """

    def __init__(
        self,
        twitter_bearer_token: Optional[str] = None,
        github_token: Optional[str] = None,
        shodan_api_key: Optional[str] = None,
        virustotal_api_key: Optional[str] = None
    ):
        self.twitter_bearer_token = twitter_bearer_token
        self.github_token = github_token
        self.shodan_api_key = shodan_api_key
        self.virustotal_api_key = virustotal_api_key

        # Initialize Twitter client
        if twitter_bearer_token:
            self.twitter_client = tweepy.Client(bearer_token=twitter_bearer_token)

    def harvest_twitter_iocs(
        self,
        keywords: List[str] = None,
        accounts: List[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Harvest IOCs from Twitter/X

        Args:
            keywords: Search keywords (e.g., ["malware", "ransomware", "CVE"])
            accounts: Threat intel Twitter accounts to monitor
            max_results: Max tweets to fetch

        Returns:
            List of extracted IOCs with context
        """
        if not self.twitter_client:
            logger.error("Twitter client not initialized")
            return []

        default_accounts = [
            'threatintel',
            'malwrhunterteam',
            'VK_Intel',
            'DailyOsint',
            'ESETresearch',
            'Unit42_Intel'
        ]

        accounts = accounts or default_accounts
        keywords = keywords or ['CVE', 'IOC', 'malware', 'threat']

        iocs = []

        try:
            # Search recent tweets
            query = ' OR '.join(keywords) + ' -is:retweet'

            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=['created_at', 'author_id', 'text', 'entities']
            )

            if not tweets.data:
                return []

            for tweet in tweets.data:
                # Extract IOCs from tweet text
                extracted = self._extract_iocs(tweet.text)

                if extracted:
                    iocs.append({
                        'source': 'twitter',
                        'tweet_id': tweet.id,
                        'created_at': tweet.created_at.isoformat(),
                        'text': tweet.text,
                        'iocs': extracted
                    })

            logger.info(f"Harvested {len(iocs)} IOCs from Twitter")
            return iocs

        except Exception as e:
            logger.error(f"Twitter harvesting failed: {e}")
            return []

    def harvest_github_secrets(
        self,
        search_terms: List[str] = None,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search GitHub for leaked credentials and secrets

        Args:
            search_terms: Search patterns (e.g., ["password", "api_key"])
            max_results: Max results

        Returns:
            List of potential secret leaks
        """
        if not self.github_token:
            logger.error("GitHub token not provided")
            return []

        default_terms = [
            'password=',
            'api_key=',
            'secret_key=',
            'aws_access_key_id',
            'private_key',
            'token='
        ]

        search_terms = search_terms or default_terms
        secrets = []

        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }

        try:
            for term in search_terms:
                url = f"https://api.github.com/search/code?q={term}&per_page=10"
                response = requests.get(url, headers=headers)

                if response.status_code == 200:
                    data = response.json()

                    for item in data.get('items', []):
                        secrets.append({
                            'source': 'github',
                            'repository': item['repository']['full_name'],
                            'path': item['path'],
                            'url': item['html_url'],
                            'search_term': term
                        })

            logger.info(f"Found {len(secrets)} potential secrets on GitHub")
            return secrets[:max_results]

        except Exception as e:
            logger.error(f"GitHub harvesting failed: {e}")
            return []

    def harvest_pastebin_dumps(
        self,
        keywords: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Monitor Pastebin for credential dumps

        Args:
            keywords: Search keywords

        Returns:
            List of relevant pastes
        """
        keywords = keywords or ['database dump', 'credentials', 'passwords', 'leak']
        dumps = []

        try:
            # Scrape recent pastes
            url = 'https://pastebin.com/archive'
            response = requests.get(url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                paste_links = soup.find_all('a', href=re.compile(r'^/\w{8}$'))

                for link in paste_links[:20]:
                    paste_id = link['href'][1:]
                    paste_url = f'https://pastebin.com/raw/{paste_id}'

                    # Fetch paste content
                    paste_response = requests.get(paste_url)

                    if paste_response.status_code == 200:
                        content = paste_response.text.lower()

                        # Check for keywords
                        if any(kw.lower() in content for kw in keywords):
                            dumps.append({
                                'source': 'pastebin',
                                'paste_id': paste_id,
                                'url': f'https://pastebin.com/{paste_id}',
                                'preview': content[:200]
                            })

            logger.info(f"Found {len(dumps)} relevant Pastebin dumps")
            return dumps

        except Exception as e:
            logger.error(f"Pastebin harvesting failed: {e}")
            return []

    def harvest_shodan_exposures(
        self,
        query: str = "port:22,3389,445",
        country: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search Shodan for exposed services

        Args:
            query: Shodan query string
            country: Filter by country code

        Returns:
            List of exposed services
        """
        if not self.shodan_api_key:
            logger.error("Shodan API key not provided")
            return []

        exposures = []

        try:
            url = f'https://api.shodan.io/shodan/host/search?key={self.shodan_api_key}&query={query}'

            if country:
                url += f'+country:{country}'

            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()

                for result in data.get('matches', []):
                    exposures.append({
                        'source': 'shodan',
                        'ip': result['ip_str'],
                        'port': result['port'],
                        'service': result.get('product', 'unknown'),
                        'version': result.get('version'),
                        'os': result.get('os'),
                        'country': result.get('location', {}).get('country_name'),
                        'org': result.get('org')
                    })

            logger.info(f"Found {len(exposures)} exposed services on Shodan")
            return exposures

        except Exception as e:
            logger.error(f"Shodan harvesting failed: {e}")
            return []

    def harvest_virustotal_iocs(
        self,
        file_hash: Optional[str] = None,
        domain: Optional[str] = None,
        ip: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query VirusTotal for malware/threat intelligence

        Args:
            file_hash: File hash (MD5, SHA1, SHA256)
            domain: Domain to check
            ip: IP address to check

        Returns:
            Threat intelligence data
        """
        if not self.virustotal_api_key:
            logger.error("VirusTotal API key not provided")
            return {}

        headers = {'x-apikey': self.virustotal_api_key}

        try:
            if file_hash:
                url = f'https://www.virustotal.com/api/v3/files/{file_hash}'
            elif domain:
                url = f'https://www.virustotal.com/api/v3/domains/{domain}'
            elif ip:
                url = f'https://www.virustotal.com/api/v3/ip_addresses/{ip}'
            else:
                return {}

            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                return {
                    'source': 'virustotal',
                    'indicator': file_hash or domain or ip,
                    'malicious': data['data']['attributes']['last_analysis_stats']['malicious'],
                    'reputation': data['data']['attributes'].get('reputation'),
                    'last_analysis_date': data['data']['attributes']['last_analysis_date']
                }

        except Exception as e:
            logger.error(f"VirusTotal query failed: {e}")

        return {}

    def _extract_iocs(self, text: str) -> Dict[str, List[str]]:
        """Extract IOCs from text using regex"""
        iocs = {
            'ip_addresses': [],
            'domains': [],
            'urls': [],
            'file_hashes': [],
            'cves': []
        }

        # IP addresses
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        iocs['ip_addresses'] = re.findall(ip_pattern, text)

        # Domains
        domain_pattern = r'\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b'
        iocs['domains'] = re.findall(domain_pattern, text.lower())

        # URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        iocs['urls'] = re.findall(url_pattern, text)

        # File hashes (MD5, SHA1, SHA256)
        hash_pattern = r'\b[a-f0-9]{32}\b|\b[a-f0-9]{40}\b|\b[a-f0-9]{64}\b'
        iocs['file_hashes'] = re.findall(hash_pattern, text.lower())

        # CVEs
        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        iocs['cves'] = re.findall(cve_pattern, text.upper())

        # Remove empty lists
        return {k: v for k, v in iocs.items() if v}


# Example usage
if __name__ == "__main__":
    harvester = OSINTHarvester(
        twitter_bearer_token="your_twitter_token",
        github_token="your_github_token",
        shodan_api_key="your_shodan_key",
        virustotal_api_key="your_vt_key"
    )

    # Harvest from Twitter
    twitter_iocs = harvester.harvest_twitter_iocs(
        keywords=['ransomware', 'CVE-2024', 'zero-day'],
        max_results=50
    )
    print(f"Twitter IOCs: {len(twitter_iocs)}")

    # Search GitHub for secrets
    github_secrets = harvester.harvest_github_secrets(
        search_terms=['aws_access_key_id', 'password='],
        max_results=20
    )
    print(f"GitHub secrets: {len(github_secrets)}")

    # Check Pastebin
    pastebin_dumps = harvester.harvest_pastebin_dumps(
        keywords=['database dump', 'credentials']
    )
    print(f"Pastebin dumps: {len(pastebin_dumps)}")

    # Shodan scan
    shodan_results = harvester.harvest_shodan_exposures(
        query='port:22 country:US'
    )
    print(f"Shodan exposures: {len(shodan_results)}")

    # VirusTotal lookup
    vt_result = harvester.harvest_virustotal_iocs(
        file_hash='44d88612fea8a8f36de82e1278abb02f'
    )
    print(f"VirusTotal: {vt_result}")
