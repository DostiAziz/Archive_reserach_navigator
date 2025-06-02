import xml.etree.ElementTree as ET
from typing import List, Dict
from tqdm import tqdm
import requests
from src.utils.logger_config import get_logger


logger = get_logger("data_pipeline")


class DataPipeline:
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"

    def search_paper(self, query: str, category: str = "all", max_results: int = 10,
                     sort_by: str = "relevance", sort_order: str = "descending") -> List[Dict]:
        """
         Searches for academic papers on an online repository using a query and optional
        filtering and sorting parameters. This function constructs a search query, sends
        an HTTP GET request to the API, and processes the response to retrieve the result
        set.

        Args:
            :param query: search query
            :param category: search category
            :param max_results: maximum number of results to return
            :param sort_by: sort by
            :param sort_order: order of sorting
        Returns:
            List[Dict]: List of search results.
        """
        try:
            search_query = f'{category}:"{query}"'
            params = {
                'search_query': search_query,
                'start': 0,
                'max_results': max_results,
                'sortBy': sort_by,
                'sortOrder': sort_order,
            }
            response = requests.get(self.base_url, params=params)
            logger.info(f"Search response status: {response.status_code}")

            if response.status_code != requests.codes.ok:
                logger.error(f"Search query failed with status code: {response.status_code}")
                raise Exception(f"Search query failed with status code: {response.status_code}")

            return self._parse_arxiv_response(response.text)
        except Exception as e:
            logger.error(f"Search query failed with exception: {e}")
            raise Exception(f"Search query failed with exception: {e}")

    def _parse_arxiv_response(self, xml_content: str) -> List[Dict]:
        """
        Parses the XML content from the arXiv API response and extracts relevant
        information about papers, such as title, abstract, DOI, publication date,
        authors, and categories.

        Args:
            xml_content (str): The XML content to parse.
        Returns:
            List[Dict]: List of search results.
        """
        try:
            root = ET.fromstring(xml_content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            papers = []

            for entry in tqdm(root.findall('atom:entry', ns), desc="Retrieving papers"):
                paper = {
                    'title': entry.find('atom:title', ns).text.strip(),
                    'abstract': entry.find('atom:summary', ns).text.strip().replace('\n', ' '),
                    'id': entry.find('atom:id', ns).text.strip(),
                    'published': entry.find('atom:published', ns).text.strip(),
                    'updated': entry.find('atom:updated', ns).text.strip(),
                }
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns).text.strip()
                    authors.append(name)

                paper['authors'] = ', '.join(authors)
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

    def list_of_queries(self, query: str, category: str = "all", max_results: int = 10, sort_by: str = "relevance",
                        sort_order: str = "descending") -> List[Dict]:
        """Process list of queries which seperated by, for retrieving papers from api
        Args:
            :param query: search queries separated by ,
            :param category: search category
            :param max_results: maximum number of results to return
            :param sort_by: sort by
            :param sort_order: order of sorting
        Returns:
            List[Dict]: List of search results.
        """
        try:
            list_of_results = []
            for query in query.split(','):
                list_of_results.extend(self.search_paper(query, category, max_results, sort_by, sort_order))
            return list_of_results
        except Exception as e:
            logger.error(f"Error parsing xml response: {e}")
            raise e


if __name__ == "__main__":
    pipeline = DataPipeline()
    results = pipeline.list_of_queries(query="RAG, graph rag", max_results=200)
    print(len(results))
    for result in results:
        print(result)
