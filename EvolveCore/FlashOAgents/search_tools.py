#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. PersonalAI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Any, Optional, Tuple
import os
import requests
import json
import time
from .tools import Tool
from .models import OpenAIServerModel

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

def read_page(url: str) -> str:
    """Read and return the content of a webpage using Jina reader."""
    jina_url = f'https://r.jina.ai/{url}'
    headers = {
        'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}',
        'X-Engine': 'browser',
        'X-Return-Format': 'markdown',
        "X-Remove-Selector": "header, .class, #id",
        "X-Retain-Images": "none",
        'X-Timeout': '10',
        'X-Token-Budget': '200000',
    }

    try:
        response = requests.get(jina_url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"Error reading page: {str(e)}"

def web_search_google_serper(
    query: str, 
    filter_year: Optional[int] = None, 
    serp_num: int = 3, 
    max_retries: int = 3
) -> Tuple[List[Dict[str, Any]], str]:
    """Perform web search using Google Serper API."""
    if not query.strip():
        return [], "Query is empty. Please provide a valid search query."
    
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "location": "United States",
        "num": serp_num
    })
    headers = {
        'X-API-KEY': os.getenv("SERPER_API_KEY"),
        'Content-Type': 'application/json'
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            results = response.json()

            if "organic" not in results or not results["organic"]:
                year_filter_msg = f" with year filter={filter_year}" if filter_year else ""
                return [], f"No results found for '{query}'{year_filter_msg}. Try a more general query."
            
            search_results = []
            for idx, page in enumerate(results["organic"], 1):
                search_results.append({
                    "idx": idx,
                    "title": page.get("title", "No title"),
                    "date": f"\nDate published: {page['date']}" if "date" in page else "",
                    "snippet": f"\n{page.get('snippet', 'No snippet')}",
                    "source": f"\nSource: {page.get('source', 'Unknown source')}",
                    "link": page.get('link', '#')
                })
            
            return search_results, ""
        
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                return [], f"Search failed after {max_retries} attempts: {str(e)}"
            time.sleep(1)
    
    return [], "Unexpected error in web search"

class WikiSearchTool(Tool):
    name = "wiki_search"
    description = "Retrieve relevant knowledge from Wikipedia and return the search results."
    inputs = {
        "query": {
            "type": "string", 
            "description": "Provide a query string for the information you want to retrieve from Wikipedia."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.tool_name = "wiki_search"

    def forward(self, query: str) -> str:
        """Execute Wikipedia search and return formatted results."""
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "exintro": True,
            "explaintext": True,
            "titles": query,
            "redirects": 1,
            "inprop": "url"
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'error' in data:
                error_info = data['error']
                return f"Wikipedia API error: {error_info.get('code', 'unknown')} - {error_info.get('info', 'unknown')}"

            pages = data.get("query", {}).get("pages", {})
            results = []
            
            for page_id, page_info in pages.items():
                if int(page_id) < 0:  # Skip invalid pages
                    continue
                    
                title = page_info.get("title", "Unknown Title")
                extract = page_info.get("extract", "No extract available")
                page_url = page_info.get("fullurl", "No URL available")
                
                results.append(
                    f"[{title}]({page_url})\n"
                    f"Summary: {extract[:500]}{'...' if len(extract) > 500 else ''}"
                )

            return "\n\n".join(results) if results else f"No relevant information found for: {query}"
        
        except requests.Timeout:
            return "Request to Wikipedia API timed out. Please try again later."
        except requests.RequestException as e:
            return f"Network error occurred: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

class WebSearchTool(Tool):
    name = "web_search"
    description = "Perform a web search query and return the search results."
    inputs = {
        "query": {
            "type": "string", 
            "description": "The web search query to perform."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.tool_name = "web_search"

    def forward(self, query: str) -> str:
        """Execute web search and return formatted results."""
        search_results, error_msg = web_search_google_serper(query, serp_num=5)
        
        if error_msg:
            return error_msg
        
        formatted_results = []
        for result in search_results:
            formatted_results.append(
                f"{result['idx']}. [{result['title']}]({result['link']})"
                f"{result['date']}{result['source']}\n"
                f"   {result['snippet'].strip()}"
            )
        
        return "\n\n".join(formatted_results) if formatted_results else "No search results found"

class CrawlPageTool(Tool):
    name = "crawl_page"
    description = "Access webpage using the provided URL and extract relevant content.  Please make full use of this tool to verify the accuracy of the searched content."
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL of the webpage to visit."
        },
        "query": {
            "type": "string",
            "description": "The specific information to extract from the webpage."
        }
    }
    output_type = "string"
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__()
        self.tool_name = "crawl_page"
        self.model = model

    @staticmethod
    def truncate_text(text: str, max_length: int = 60000) -> str:
        """Truncate text to specified length."""
        return text if len(text) <= max_length else text[:max_length] + "...(truncated)"

    def get_summary_prompt(self, query: str, url: str, content: str) -> str:
        """Generate prompt for content summarization."""
        return (
            f"Task: Extract all content from the web page that matches the search query.\n"
            f"Search Query: {query}\n\n"
            f"Web Page Content [url:{url}]:\n{content}\n\n"
            "Instructions:\n"
            "- Summarize all relevant content for the query (text, tables, lists) into concise points\n"
            "- If no relevant information exists, please straightly output 'No relevant information'\n"
            "- Keep the summary under 500 words"
        )

    def retry_predict(self, prompt: str, max_retries: int = 3) -> str:
        """Retry model prediction with exponential backoff."""
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(max_retries):
            try:
                response = self.model(messages)
                if hasattr(response, 'content'):
                    content = response.content
                    return content.strip() if isinstance(content, str) else str(content)
                return str(response)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Content extraction failed: {str(e)}"
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        
        return "Content extraction failed after multiple attempts"

    def forward(self, url: str, query: str) -> str:
        """Crawl webpage and extract relevant content."""
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            return "Invalid URL format. Must start with http:// or https://"
        
        page_content = read_page(url)
        if page_content.startswith("Error"):
            return page_content
        
        truncated_content = self.truncate_text(page_content)
        prompt = self.get_summary_prompt(query, url, truncated_content)
        
        return self.retry_predict(prompt)
    
__all__ = [
    "WikiSearchTool",
    "WebSearchTool",
    "CrawlPageTool",
]