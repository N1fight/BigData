"""
Модуль полнотекстового поиска
"""

import math
from typing import List, Dict, Tuple, Set, Any
import logging
from collections import defaultdict
from database import Database
from config import SEARCH_CONFIG, STOP_WORDS
from utils import clean_text, tokenize, generate_snippet, logger


class SearchEngine:
    """Класс полнотекстового поиска"""

    def __init__(self, db: Database):
        self.db = db
        self.total_documents = self.db.get_total_documents()
        self.results_per_page = SEARCH_CONFIG['results_per_page']
        self.snippet_length = SEARCH_CONFIG['snippet_length']

        # Кэш для IDF значений
        self.idf_cache = {}

        logger.info(f"SearchEngine initialized for {self.total_documents} documents")

    def calculate_idf(self, word: str) -> float:
        """
        Расчет IDF (Inverse Document Frequency)
        """
        # Используем кэш если уже считали
        if word in self.idf_cache:
            return self.idf_cache[word]

        # Количество документов, содержащих слово
        docs_with_word = self.db.get_documents_for_word(word)
        doc_count = len(docs_with_word)

        if doc_count == 0:
            self.idf_cache[word] = 0.0
            return 0.0

        # Формула IDF с smoothing
        idf = math.log((self.total_documents + 1) / (doc_count + 1)) + 1
        self.idf_cache[word] = idf
        return idf

    def calculate_tf(self, term_count: int, total_terms: int) -> float:
        """
        Расчет TF (Term Frequency) с нормализацией
        """
        if total_terms == 0:
            return 0.0

        # Простая TF формула (можно использовать log или другие варианты)
        return term_count / total_terms

    def search_term_at_a_time(self, query: str, use_pagerank: bool = True) -> List[Tuple[int, float, str]]:
        """
        Поиск с использованием подхода Term-at-a-Time
        """
        # Очистка и токенизация запроса
        query_terms = tokenize(query, STOP_WORDS)

        if not query_terms:
            return []

        # Словарь для хранения оценок документов
        scores = defaultdict(float)
        doc_lengths = {}  # Для хранения длин документов в терминах

        # Получаем все документы для предварительной обработки
        all_docs = self.db.get_all_documents()

        # Собираем информацию о длинах документов
        for doc_id, url, title in all_docs:
            content = self.db.get_document_content(doc_id)
            if content:
                tokens = tokenize(content, STOP_WORDS)
                doc_lengths[doc_id] = len(tokens)

        # Обработка каждого термина отдельно
        for term in query_terms:
            # Получение документов, содержащих термин
            docs_with_term = self.db.get_documents_for_word(term)

            # Расчет IDF для термина
            idf = self.calculate_idf(term)

            if idf == 0.0:
                continue  # Пропускаем слова, которых нет в коллекции

            # Добавление вклада термина в оценку каждого документа
            for doc_id, raw_tf in docs_with_term:
                if doc_id not in doc_lengths or doc_lengths[doc_id] == 0:
                    continue

                # Нормализованный TF
                term_count = int(raw_tf * doc_lengths[doc_id])  # Преобразуем обратно в count
                tf = self.calculate_tf(term_count, doc_lengths[doc_id])

                # TF-IDF оценка
                tfidf = tf * idf
                scores[doc_id] += tfidf

        # Применение PageRank, если требуется
        if use_pagerank and scores:
            pageranks = self.db.get_all_pageranks()

            # Нормализуем TF-IDF scores
            if scores:
                max_score = max(scores.values())
                if max_score > 0:
                    for doc_id in scores:
                        scores[doc_id] = scores[doc_id] / max_score

            # Применяем PageRank
            for doc_id in list(scores.keys()):
                pagerank = pageranks.get(doc_id, 1.0)
                # Комбинирование: TF-IDF * (1 + PageRank)
                # PageRank обычно в диапазоне 0-1, поэтому добавляем 1
                scores[doc_id] = scores[doc_id] * (1.0 + pagerank)

        # Сортировка результатов
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Форматирование результатов
        formatted_results = []
        for doc_id, score in sorted_results[:self.results_per_page]:
            content = self.db.get_document_content(doc_id)
            if content:
                snippet = generate_snippet(content, query_terms, self.snippet_length)
                formatted_results.append((doc_id, score, snippet))

        return formatted_results

    def search_document_at_a_time(self, query: str, use_pagerank: bool = True) -> List[Tuple[int, float, str]]:
        """
        Поиск с использованием подхода Document-at-a-Time
        """
        # Очистка и токенизация запроса
        query_terms = tokenize(query, STOP_WORDS)

        if not query_terms:
            return []

        # Предварительный расчет IDF для всех терминов
        term_idf = {}
        for term in query_terms:
            term_idf[term] = self.calculate_idf(term)

        # Словарь для хранения TF-IDF значений для каждого документа
        doc_scores = {}

        # Получение всех документов
        all_docs = self.db.get_all_documents()

        # Обработка каждого документа отдельно
        for doc_id, url, title in all_docs:
            # Получение содержимого документа
            content = self.db.get_document_content(doc_id)
            if not content:
                continue

            # Токенизация содержимого
            doc_tokens = tokenize(content, STOP_WORDS)
            total_terms = len(doc_tokens)

            if total_terms == 0:
                continue

            # Расчет TF для каждого термина запроса в этом документе
            doc_score = 0.0
            for term in query_terms:
                # Пропускаем термины с нулевым IDF
                if term_idf[term] == 0.0:
                    continue

                # Подсчет частоты термина в документе
                term_count = doc_tokens.count(term)
                if term_count > 0:
                    tf = self.calculate_tf(term_count, total_terms)
                    idf = term_idf[term]
                    doc_score += tf * idf

            if doc_score > 0:
                doc_scores[doc_id] = doc_score

        # Применение PageRank, если требуется
        if use_pagerank and doc_scores:
            pageranks = self.db.get_all_pageranks()

            # Нормализуем scores
            max_score = max(doc_scores.values())
            if max_score > 0:
                for doc_id in doc_scores:
                    doc_scores[doc_id] = doc_scores[doc_id] / max_score

            # Применяем PageRank
            for doc_id in list(doc_scores.keys()):
                pagerank = pageranks.get(doc_id, 1.0)
                doc_scores[doc_id] = doc_scores[doc_id] * (1.0 + pagerank)

        # Сортировка результатов
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Форматирование результатов
        formatted_results = []
        for doc_id, score in sorted_results[:self.results_per_page]:
            content = self.db.get_document_content(doc_id)
            if content:
                snippet = generate_snippet(content, query_terms, self.snippet_length)
                formatted_results.append((doc_id, score, snippet))

        return formatted_results

    def search(self, query: str, method: str = 'term', use_pagerank: bool = True) -> List[Tuple[int, float, str]]:
        """
        Основной метод поиска
        """
        logger.info(f"Searching for: '{query}' (method: {method}, pagerank: {use_pagerank})")

        if method.lower() == 'term':
            results = self.search_term_at_a_time(query, use_pagerank)
        elif method.lower() == 'document':
            results = self.search_document_at_a_time(query, use_pagerank)
        else:
            logger.warning(f"Unknown search method: {method}, using 'term'")
            results = self.search_term_at_a_time(query, use_pagerank)

        logger.info(f"Found {len(results)} results for query: '{query}'")
        return results

    def print_results(self, query: str, method: str = 'term', use_pagerank: bool = True):
        """
        Поиск и вывод результатов
        """
        results = self.search(query, method, use_pagerank)

        print(f"\n=== Search Results for: '{query}' ===")
        print(f"Method: {method}-at-a-time, PageRank: {use_pagerank}")
        print(f"Found {len(results)} results")
        print("-" * 80)

        for i, (doc_id, score, snippet) in enumerate(results, 1):
            # Получаем информацию о документе
            doc_info = self.db.get_document_info(doc_id)
            title = doc_info[1] if doc_info else f"Document {doc_id}"
            url = doc_info[0] if doc_info else "Unknown URL"

            print(f"\n{i}. Document ID: {doc_id}")
            print(f"   Title: {title}")
            print(f"   URL: {url}")
            print(f"   Score: {score:.6f}")  # Увеличиваем точность до 6 знаков
            print(f"   Snippet: {snippet}")

        if not results:
            print("\nNo results found. Try different search terms.")

        print("\n" + "=" * 80)