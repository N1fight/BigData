"""
Реализация PageRank с использованием MapReduce парадигмы
"""

import sqlite3
from typing import List, Dict, Tuple, Any
import logging
from database import Database
from config import PAGERANK_CONFIG
from utils import logger


class PageRankMapReduce:
    """Класс для вычисления PageRank с использованием MapReduce"""

    def __init__(self, db: Database):
        self.db = db
        self.damping_factor = PAGERANK_CONFIG['damping_factor']
        self.max_iterations = PAGERANK_CONFIG['max_iterations']
        self.tolerance = PAGERANK_CONFIG['tolerance']

        # Получение информации о документах
        self.documents = self.db.get_all_documents()
        self.doc_ids = [doc[0] for doc in self.documents]
        self.num_documents = len(self.doc_ids)

        # Инициализация PageRank
        self.pagerank = {doc_id: 1.0 / self.num_documents for doc_id in self.doc_ids}

        # Построение графа ссылок
        self.outgoing_links = {}
        self.incoming_links = {}

        for doc_id in self.doc_ids:
            outgoing = self.db.get_outgoing_links(doc_id)
            self.outgoing_links[doc_id] = [link[0] for link in outgoing if link[0]]

            incoming = self.db.get_incoming_links(doc_id)
            self.incoming_links[doc_id] = incoming

        logger.info(f"PageRankMapReduce initialized for {self.num_documents} documents")

    def map_phase(self, doc_id: int, pagerank: float) -> List[Tuple[int, float]]:
        """
        Map фаза: распределение PageRank по исходящим ссылкам
        """
        if doc_id not in self.outgoing_links:
            return []

        outgoing = self.outgoing_links[doc_id]
        if not outgoing:
            return []

        # Равномерное распределение PageRank
        share = pagerank / len(outgoing)

        # Возвращаем пары (целевой документ, доля PageRank)
        return [(target_id, share) for target_id in outgoing]

    def reduce_phase(self, contributions: List[Tuple[int, float]]) -> Dict[int, float]:
        """
        Reduce фаза: суммирование входящих PageRank
        """
        reduced = {}
        for doc_id, contribution in contributions:
            reduced[doc_id] = reduced.get(doc_id, 0.0) + contribution

        return reduced

    def calculate_pagerank_iteration(self, current_pagerank: Dict[int, float]) -> Dict[int, float]:
        """
        Вычисление одной итерации PageRank
        """
        # Map фаза: сбор всех contributions
        all_contributions = []
        for doc_id, rank in current_pagerank.items():
            contributions = self.map_phase(doc_id, rank)
            all_contributions.extend(contributions)

        # Reduce фаза: суммирование contributions
        new_ranks = self.reduce_phase(all_contributions)

        # Применение формулы PageRank
        for doc_id in self.doc_ids:
            # Базовая часть (для документов без входящих ссылок)
            base_rank = (1 - self.damping_factor) / self.num_documents

            # Часть от входящих ссылок
            incoming_rank = new_ranks.get(doc_id, 0.0) * self.damping_factor

            # Суммирование
            new_ranks[doc_id] = base_rank + incoming_rank

        return new_ranks

    def calculate(self) -> Dict[int, float]:
        """
        Основной метод вычисления PageRank
        """
        logger.info("Starting PageRank calculation using MapReduce")

        current_pagerank = self.pagerank.copy()

        for iteration in range(self.max_iterations):
            # Вычисление новой итерации
            new_pagerank = self.calculate_pagerank_iteration(current_pagerank)

            # Проверка сходимости
            convergence = self.calculate_convergence(current_pagerank, new_pagerank)

            # Обновление текущих значений
            current_pagerank = new_pagerank

            # Сохранение в базу данных
            for doc_id, rank in current_pagerank.items():
                self.db.update_pagerank(doc_id, rank, iteration + 1)

            logger.info(f"Iteration {iteration + 1}: Convergence = {convergence:.6f}")

            # Проверка условия остановки
            if convergence < self.tolerance:
                logger.info(f"PageRank converged after {iteration + 1} iterations")
                break

        self.pagerank = current_pagerank
        return self.pagerank

    def calculate_convergence(self, old_ranks: Dict[int, float],
                              new_ranks: Dict[int, float]) -> float:
        """
        Вычисление сходимости (среднеквадратичная ошибка)
        """
        total_diff = 0.0
        for doc_id in self.doc_ids:
            diff = new_ranks.get(doc_id, 0.0) - old_ranks.get(doc_id, 0.0)
            total_diff += diff * diff

        return (total_diff / self.num_documents) ** 0.5

    def get_top_documents(self, n: int = 10) -> List[Tuple[int, float, str]]:
        """
        Получение топ-N документов по PageRank
        """
        # Получение информации о документах
        documents_info = {}
        for doc_id, url, title in self.documents:
            documents_info[doc_id] = (url, title)

        # Сортировка по PageRank
        sorted_docs = sorted(self.pagerank.items(),
                             key=lambda x: x[1],
                             reverse=True)[:n]

        # Формирование результата
        result = []
        for doc_id, rank in sorted_docs:
            url, title = documents_info.get(doc_id, ("Unknown", "Unknown"))
            result.append((doc_id, rank, url, title[:50]))

        return result

    def print_statistics(self):
        """Вывод статистики PageRank"""
        print("\n=== PageRank Statistics (MapReduce) ===")
        print(f"Total documents: {self.num_documents}")
        print(f"Damping factor: {self.damping_factor}")
        print(f"Max iterations: {self.max_iterations}")

        top_docs = self.get_top_documents(5)
        print("\nTop 5 documents by PageRank:")
        for doc_id, rank, url, title in top_docs:
            print(f"  ID: {doc_id}, Rank: {rank:.6f}")
            print(f"  URL: {url}")
            print(f"  Title: {title}")
            print()