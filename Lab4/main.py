"""
Главный файл для запуска поискового движка
"""

import argparse
import os
from typing import List, Dict, Any
import logging
from config import START_URLS
from database import Database
from parser import WebParser
from pagerank_mapreduce import PageRankMapReduce
from pagerank_pregel import PageRankPregel
from search_engine import SearchEngine
from utils import logger


class SearchEngineApp:
    """Главный класс приложения поискового движка"""

    def __init__(self):
        self.db = Database()
        self.parser = None
        self.mapreduce_pr = None
        self.pregel_pr = None
        self.search_engine = None

        logger.info("Search Engine Application initialized")

    def initialize_components(self):
        """Инициализация компонентов после добавления документов"""
        self.parser = WebParser(self.db)
        self.search_engine = SearchEngine(self.db)

    def crawl_websites(self, start_urls: List[str] = None, max_pages: int = 50):
        """Парсинг веб-сайтов"""
        if start_urls is None:
            start_urls = START_URLS

        logger.info(f"Starting crawling with {len(start_urls)} start URLs")

        if not self.parser:
            self.parser = WebParser(self.db)

        self.parser.crawl(start_urls, max_pages)

        # Инициализация поискового движка после парсинга
        self.search_engine = SearchEngine(self.db)

        # Вывод статистики
        stats = self.parser.get_statistics()
        print("\n=== Crawling Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")

        return stats

    def test_with_local_data(self):
        """Тестирование с локальными данными (без интернета)"""
        print("\n=== Testing with Local Data ===")

        # Очищаем базу данных
        self.db.clear_database()

        # Создаем тестовые документы
        test_documents = [
            {
                'url': 'http://example.com/doc1',
                'title': 'Document 1: Machine Learning',
                'content': 'Machine learning is a subset of artificial intelligence. Machine learning algorithms build models based on training data.',
                'links': ['http://example.com/doc2', 'http://example.com/doc3']
            },
            {
                'url': 'http://example.com/doc2',
                'title': 'Document 2: Artificial Intelligence',
                'content': 'Artificial intelligence is the simulation of human intelligence by machines. AI includes machine learning and deep learning.',
                'links': ['http://example.com/doc1', 'http://example.com/doc3']
            },
            {
                'url': 'http://example.com/doc3',
                'title': 'Document 3: Data Science',
                'content': 'Data science combines statistics, machine learning, and data analysis. Data scientists use machine learning for predictions.',
                'links': ['http://example.com/doc1']
            },
            {
                'url': 'http://example.com/doc4',
                'title': 'Document 4: Deep Learning',
                'content': 'Deep learning is a type of machine learning based on neural networks. Deep learning requires large amounts of data.',
                'links': ['http://example.com/doc2']
            }
        ]

        # Добавляем документы в базу
        print("\nAdding test documents to database...")
        doc_ids = {}

        for doc in test_documents:
            doc_id = self.db.add_document(doc['url'], doc['title'], doc['content'])
            doc_ids[doc['url']] = doc_id
            print(f"  Added: {doc['title']} (ID: {doc_id})")

        # Добавляем ссылки между документами
        print("\nAdding links between documents...")
        for doc in test_documents:
            source_id = doc_ids[doc['url']]
            for link in doc['links']:
                target_id = doc_ids.get(link)
                self.db.add_link(source_id, link, target_id)
                print(f"  Link: {source_id} -> {link} (target ID: {target_id})")

        # Индексируем слова
        print("\nIndexing words...")
        from utils import tokenize
        from config import STOP_WORDS

        for doc in test_documents:
            doc_id = doc_ids[doc['url']]
            tokens = tokenize(doc['content'], STOP_WORDS)

            # Расчет TF
            from collections import Counter
            total_terms = len(tokens)
            term_counts = Counter(tokens)

            # Добавление в обратный индекс
            for word, count in term_counts.items():
                word_id = self.db.add_word(word)
                if word_id != -1:
                    tf = count / total_terms if total_terms > 0 else 0
                    # Используем простые позиции для демонстрации
                    positions = list(range(count))
                    self.db.add_inverted_index_entry(word_id, doc_id, tf, positions)

        print(f"\nTotal documents in database: {self.db.get_total_documents()}")

        # Инициализируем компоненты
        self.initialize_components()

        return True

    def calculate_pagerank_mapreduce(self):
        """Вычисление PageRank с использованием MapReduce"""
        if self.db.get_total_documents() == 0:
            logger.warning("No documents in database.")
            return {}

        print("\n=== Calculating PageRank (MapReduce) ===")
        self.mapreduce_pr = PageRankMapReduce(self.db)
        pagerank = self.mapreduce_pr.calculate()
        self.mapreduce_pr.print_statistics()

        return pagerank

    def calculate_pagerank_pregel(self):
        """Вычисление PageRank с использованием Pregel"""
        if self.db.get_total_documents() == 0:
            logger.warning("No documents in database.")
            return {}

        print("\n=== Calculating PageRank (Pregel) ===")
        self.pregel_pr = PageRankPregel(self.db)
        pagerank = self.pregel_pr.calculate()
        self.pregel_pr.print_statistics()

        return pagerank

    def search(self, query: str, method: str = 'term', use_pagerank: bool = True):
        """Выполнение поиска"""
        if self.db.get_total_documents() == 0:
            logger.warning("No documents in database.")
            return []

        if not self.search_engine:
            self.search_engine = SearchEngine(self.db)

        return self.search_engine.search(query, method, use_pagerank)

    def interactive_search(self):
        """Интерактивный режим поиска"""
        if self.db.get_total_documents() == 0:
            print("No documents in database. Please add documents first.")
            return

        if not self.search_engine:
            self.search_engine = SearchEngine(self.db)

        print("\n" + "=" * 60)
        print("  SEARCH ENGINE - Interactive Mode")
        print("=" * 60)

        while True:
            print("\nOptions:")
            print("  1. Search (Term-at-a-Time with PageRank)")
            print("  2. Search (Document-at-a-Time with PageRank)")
            print("  3. Search without PageRank")
            print("  4. Compare search methods")
            print("  5. Show database statistics")
            print("  6. Calculate PageRank (MapReduce)")
            print("  7. Calculate PageRank (Pregel)")
            print("  8. Exit")

            choice = input("\nEnter your choice (1-8): ").strip()

            if choice == '1':
                query = input("Enter search query: ").strip()
                if query:
                    self.search_engine.print_results(query, 'term', True)

            elif choice == '2':
                query = input("Enter search query: ").strip()
                if query:
                    self.search_engine.print_results(query, 'document', True)

            elif choice == '3':
                query = input("Enter search query: ").strip()
                if query:
                    self.search_engine.print_results(query, 'term', False)

            elif choice == '4':
                query = input("Enter search query for comparison: ").strip()
                if query:
                    print("\n=== Comparing Search Methods ===")

                    # Term-at-a-Time с PageRank
                    print("\n1. Term-at-a-Time with PageRank:")
                    results1 = self.search(query, 'term', True)
                    for i, (doc_id, score, snippet) in enumerate(results1[:3], 1):
                        doc_info = self.db.get_document_info(doc_id)
                        title = doc_info[1][:30] + "..." if doc_info and len(doc_info[1]) > 30 else (
                            doc_info[1] if doc_info else "Unknown")
                        print(f"   {i}. Doc {doc_id} ({title}), Score: {score:.4f}")

                    # Document-at-a-Time с PageRank
                    print("\n2. Document-at-a-Time with PageRank:")
                    results2 = self.search(query, 'document', True)
                    for i, (doc_id, score, snippet) in enumerate(results2[:3], 1):
                        doc_info = self.db.get_document_info(doc_id)
                        title = doc_info[1][:30] + "..." if doc_info and len(doc_info[1]) > 30 else (
                            doc_info[1] if doc_info else "Unknown")
                        print(f"   {i}. Doc {doc_id} ({title}), Score: {score:.4f}")

                    # Term-at-a-Time без PageRank
                    print("\n3. Term-at-a-Time without PageRank:")
                    results3 = self.search(query, 'term', False)
                    for i, (doc_id, score, snippet) in enumerate(results3[:3], 1):
                        doc_info = self.db.get_document_info(doc_id)
                        title = doc_info[1][:30] + "..." if doc_info and len(doc_info[1]) > 30 else (
                            doc_info[1] if doc_info else "Unknown")
                        print(f"   {i}. Doc {doc_id} ({title}), Score: {score:.4f}")

            elif choice == '5':
                self.show_statistics()

            elif choice == '6':
                self.calculate_pagerank_mapreduce()

            elif choice == '7':
                self.calculate_pagerank_pregel()

            elif choice == '8':
                print("Exiting interactive mode.")
                break

            else:
                print("Invalid choice. Please try again.")

    def show_statistics(self):
        """Показать статистику базы данных"""
        print("\n=== Database Statistics ===")
        total_docs = self.db.get_total_documents()
        print(f"Total documents: {total_docs}")

        if total_docs == 0:
            print("No documents in database.")
            return

        # Получение примеров документов
        documents = self.db.get_all_documents()[:5]
        print(f"\nFirst {len(documents)} documents:")
        for doc_id, url, title in documents:
            print(f"  ID: {doc_id}, Title: {title[:50]}...")

        # Получение PageRank статистики
        pageranks = self.db.get_all_pageranks()
        if pageranks:
            sorted_pr = sorted(pageranks.items(), key=lambda x: x[1], reverse=True)[:3]
            print("\nTop 3 documents by PageRank:")
            for doc_id, rank in sorted_pr:
                doc_info = self.db.get_document_info(doc_id)
                title = doc_info[1][:50] + "..." if doc_info and len(doc_info[1]) > 50 else (
                    doc_info[1] if doc_info else "Unknown")
                print(f"  ID: {doc_id}, Rank: {rank:.6f}, Title: {title}")
        else:
            print("\nPageRank not calculated yet. Use options 6 or 7 to calculate.")

    def run_demo(self):
        """Запуск демонстрации всех функций"""
        print("=" * 70)
        print("SEARCH ENGINE DEMONSTRATION")
        print("=" * 70)

        # Шаг 1: Тестирование с локальными данными
        print("\n1. Setting up with test data...")
        self.test_with_local_data()

        # Шаг 2: PageRank MapReduce
        print("\n2. Calculating PageRank using MapReduce...")
        self.calculate_pagerank_mapreduce()

        # Шаг 3: PageRank Pregel
        print("\n3. Calculating PageRank using Pregel-like approach...")
        self.calculate_pagerank_pregel()

        # Шаг 4: Поиск
        print("\n4. Testing search functionality...")

        test_queries = ["machine learning", "artificial intelligence", "data science"]

        for query in test_queries:
            print(f"\nQuery: '{query}'")

            # Term-at-a-Time
            print("  Term-at-a-Time with PageRank (first 2 results):")
            results = self.search(query, 'term', True)
            for i, (doc_id, score, snippet) in enumerate(results[:2], 1):
                doc_info = self.db.get_document_info(doc_id)
                title = doc_info[1] if doc_info else "Unknown"
                print(f"    {i}. Doc {doc_id}: {title[:40]}...")
                print(f"       Score: {score:.4f}, Snippet: {snippet[:60]}...")

            # Document-at-a-Time
            print("  Document-at-a-Time with PageRank (first 2 results):")
            results = self.search(query, 'document', True)
            for i, (doc_id, score, snippet) in enumerate(results[:2], 1):
                doc_info = self.db.get_document_info(doc_id)
                title = doc_info[1] if doc_info else "Unknown"
                print(f"    {i}. Doc {doc_id}: {title[:40]}...")
                print(f"       Score: {score:.4f}, Snippet: {snippet[:60]}...")

        # Шаг 5: Интерактивный режим
        print("\n5. Starting interactive search mode...")
        print("   (Try queries like 'machine', 'learning', 'deep', 'neural')")
        self.interactive_search()

    def run_real_crawl_demo(self):
        """Демонстрация с реальным парсингом (требует интернет)"""
        print("=" * 70)
        print("REAL WEB CRAWLING DEMONSTRATION")
        print("=" * 70)
        print("Note: This requires internet connection and will parse real Wikipedia pages")

        confirm = input("\nContinue with real web crawling? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Skipping real crawling demo.")
            return

        # Шаг 1: Парсинг реальных страниц
        print("\n1. Crawling Wikipedia pages...")
        self.crawl_websites(START_URLS[:3], max_pages=15)

        # Шаг 2: PageRank
        print("\n2. Calculating PageRank...")
        self.calculate_pagerank_mapreduce()

        # Шаг 3: Поиск
        print("\n3. Testing search...")

        test_queries = ["science", "learning", "web", "search"]

        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = self.search(query, 'term', True)
            print(f"  Found {len(results)} results")

            for i, (doc_id, score, snippet) in enumerate(results[:2], 1):
                doc_info = self.db.get_document_info(doc_id)
                title = doc_info[1] if doc_info else "Unknown"
                print(f"  {i}. {title[:50]}...")
                print(f"     Score: {score:.4f}")

    def cleanup(self):
        """Очистка ресурсов"""
        self.db.close()
        logger.info("Application cleanup completed")


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Search Engine Application')
    parser.add_argument('--demo', action='store_true', help='Run full demonstration with test data')
    parser.add_argument('--real-demo', action='store_true',
                        help='Run demonstration with real web crawling (requires internet)')
    parser.add_argument('--crawl', action='store_true', help='Crawl websites (requires internet)')
    parser.add_argument('--test-data', action='store_true', help='Load test data without internet')
    parser.add_argument('--pagerank', choices=['mapreduce', 'pregel', 'both'],
                        help='Calculate PageRank')
    parser.add_argument('--search', type=str, help='Search query')
    parser.add_argument('--method', choices=['term', 'document'], default='term',
                        help='Search method (term-at-a-time or document-at-a-time)')
    parser.add_argument('--no-pagerank', action='store_true',
                        help='Disable PageRank in search')
    parser.add_argument('--interactive', action='store_true',
                        help='Start interactive mode')

    args = parser.parse_args()

    app = SearchEngineApp()

    try:
        if args.demo:
            app.run_demo()

        elif args.real_demo:
            app.run_real_crawl_demo()

        elif args.crawl:
            app.crawl_websites()

        elif args.test_data:
            app.test_with_local_data()
            print("\nTest data loaded successfully!")
            print(f"Total documents: {app.db.get_total_documents()}")

        elif args.pagerank:
            # Сначала загружаем тестовые данные, если база пуста
            if app.db.get_total_documents() == 0:
                print("Database is empty. Loading test data first...")
                app.test_with_local_data()

            if args.pagerank in ['mapreduce', 'both']:
                app.calculate_pagerank_mapreduce()
            if args.pagerank in ['pregel', 'both']:
                app.calculate_pagerank_pregel()

        elif args.search:
            if app.db.get_total_documents() == 0:
                print("Database is empty. Loading test data first...")
                app.test_with_local_data()
                app.initialize_components()

            use_pagerank = not args.no_pagerank
            results = app.search(args.search, args.method, use_pagerank)

            print(f"\nSearch results for '{args.search}':")
            if results:
                for i, (doc_id, score, snippet) in enumerate(results, 1):
                    doc_info = app.db.get_document_info(doc_id)
                    title = doc_info[1] if doc_info else "Unknown"
                    url = doc_info[0] if doc_info else "Unknown"

                    print(f"\n{i}. Document ID: {doc_id}")
                    print(f"   Title: {title}")
                    print(f"   URL: {url}")
                    print(f"   Score: {score:.4f}")
                    print(f"   Snippet: {snippet}")
            else:
                print("No results found.")

        elif args.interactive:
            if app.db.get_total_documents() == 0:
                print("Database is empty. Loading test data first...")
                app.test_with_local_data()

            app.interactive_search()

        else:
            # Если не указаны аргументы, запускаем демо с тестовыми данными
            print("No arguments provided. Running demonstration with test data...")
            app.run_demo()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        app.cleanup()


if __name__ == "__main__":

    main()
