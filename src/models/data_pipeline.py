import requests
import pandas as pd
import xml.etree.ElementTree as ET
from typing import List, Dict
import time
import logging

logger = logging.getLogger(__name__)


class DataPipeline:
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"

    def search_paper(self, query: str, category: str = "all", max_results: int = 10,
                     sort_by: str = "relevance", sort_order: str = "desc") -> List[Dict]:
        """This function is used to search a paper by querying the ArXiv API.
        Args:

            :param query: query to search for
            :param category: category to search for (all:all, au:author, ti:title, ab:abstract, etc..)
            :param max_results:
            :param sort_by:
            :param sort_order:
         Returns:
            List[Dict]: A list of dictionaries of search results.
        """

        search_query = f"{category}={query}"

        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sort': sort_by,
            'sorOrder': sort_order,
        }

        logger.info(f"Search query: {search_query}")
        response = requests.get(self.base_url, params=params)

        if response.status_code != requests.codes.ok:
            logger.error(f"Search query failed with status code: {response.status_code}")
            raise Exception(f"Search query failed with status code: {response.status_code}")

        return self._parse_arxiv_response(response.text)

    def _parse_arxiv_response(self, xml_content: str) -> List[Dict]:
        """Parse arXiv response into structured data.
        Args:
            :param response: response from ArXiv API
        Returns:
        List[Dict]: A list of dictionaries of search results.

        """
        try:

            root = ET.fromstring(xml_content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            papers = []
            for entry in root.findall('atom:entry', ns):
                paper = {}
                # Extract title, abstract, dio and published
                paper['title'] = entry.find('atom:title', ns).text.strip()
                paper['abstract'] = entry.find('atom:summary', ns).text.strip()
                paper['doi'] = entry.find('atom:id', ns).text.strip()
                paper['published'] = entry.find('atom:published', ns).text.strip()

                # Extract author names
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns).text.strip()
                    authors.append(name)
                paper['authors'] = ', '.join(authors)
                # Extract categories
                categories = []
                for category in entry.findall('atom:category', ns):
                    categories.append(category.get('term'))
                paper['categories'] = ', '.join(categories)

                papers.append(paper)
            logger.info(f"Found {len(papers)} papers")
            return papers
        except Exception as e:
            logger.error(f"Error parsing xml response: {e}")
            raise e
