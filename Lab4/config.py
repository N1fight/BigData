"""
Конфигурация поискового движка
"""

import os
from typing import List, Dict, Any

# Конфигурация парсера
PARSER_CONFIG = {
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'timeout': 10,
    'max_pages': 50,  # Максимальное количество страниц для парсинга
    'max_content_length': 1000000,  # Максимальный размер контента в байтах
}

# Список начальных URL для парсинга (примеры)
START_URLS = [
    'https://en.wikipedia.org/wiki/Data_science',
    'https://en.wikipedia.org/wiki/Machine_learning',
    'https://en.wikipedia.org/wiki/Artificial_intelligence',
    'https://en.wikipedia.org/wiki/Web_crawler',
    'https://en.wikipedia.org/wiki/Search_engine',
]

# Конфигурация базы данных
DATABASE_CONFIG = {
    'db_name': 'search_engine.db',
    'documents_table': 'documents',
    'words_table': 'words',
    'links_table': 'links',
    'inverted_index_table': 'inverted_index',
    'pagerank_table': 'pagerank',
}

# Конфигурация PageRank
PAGERANK_CONFIG = {
    'damping_factor': 0.85,
    'max_iterations': 100,
    'tolerance': 1e-6,
}

# Конфигурация поиска
SEARCH_CONFIG = {
    'results_per_page': 10,
    'snippet_length': 150,
}

# Список стоп-слов (упрощенный)
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
    'should', 'may', 'might', 'must', 'can', 'could', 'i', 'you', 'he',
    'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'this',
    'that', 'these', 'those', 'am', 'is', 'are'
}