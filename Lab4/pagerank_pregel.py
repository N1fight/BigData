# -*- coding: utf-8 -*-
"""
Реализация PageRank с использованием упрощенного Pregel-like подхода
"""

from typing import List, Dict, Tuple, Any
import logging
from database import Database
from config import PAGERANK_CONFIG
from utils import logger


class PregelVertex:
    """Вершина графа для упрощенной Pregel реализации"""

    def __init__(self, vertex_id: int):
        self.id = vertex_id
        self.value = 0.0  # Текущее значение PageRank
        self.outgoing_edges = []  # Исходящие ребра
        self.active = True

    def send_messages(self) -> List[Tuple[int, float]]:
        """
        Отправка сообщений соседям
        """
        if not self.outgoing_edges:
            return []

        # Равномерное распределение PageRank по исходящим ссылкам
        share = self.value / len(self.outgoing_edges)
        return [(neighbor, share) for neighbor in self.outgoing_edges]


class SimplePregelGraph:
    """Упрощенный граф для Pregel-like реализации PageRank"""

    def __init__(self, num_vertices: int):
        self.vertices = [PregelVertex(i) for i in range(num_vertices)]
        self.num_vertices = num_vertices
        self.superstep = 0
        self.messages = {}

    def add_edge(self, source: int, target: int):
        """Добавление ребра в граф"""
        if 0 <= source < self.num_vertices and 0 <= target < self.num_vertices:
            self.vertices[source].outgoing_edges.append(target)

    def initialize_pagerank(self):
        """Инициализация значений PageRank"""
        initial_value = 1.0 / self.num_vertices
        for vertex in self.vertices:
            vertex.value = initial_value
            vertex.active = True

    def run_superstep(self, damping_factor: float) -> bool:
        """
        Выполнение одного суперстепа
        Возвращает True, если есть активные вершины
        """
        self.superstep += 1
        self.messages = {}

        # Фаза 1: Отправка сообщений активными вершинами
        for vertex in self.vertices:
            if vertex.active:
                messages = vertex.send_messages()
                for target_id, message_value in messages:
                    if target_id not in self.messages:
                        self.messages[target_id] = []
                    self.messages[target_id].append(message_value)

        # Фаза 2: Получение сообщений и обновление значений
        active_vertices = 0

        for vertex in self.vertices:
            # Получение входящих сообщений
            incoming_messages = self.messages.get(vertex.id, [])
            incoming_sum = sum(incoming_messages) if incoming_messages else 0

            # Вычисление нового значения PageRank
            random_walk = (1 - damping_factor) / self.num_vertices
            new_value = random_walk + damping_factor * incoming_sum

            # Проверка на сходимость
            if abs(new_value - vertex.value) > 1e-10:
                vertex.active = True
                active_vertices += 1
            else:
                vertex.active = False

            # Обновление значения
            vertex.value = new_value

        return active_vertices > 0


class PageRankPregel:
    """Класс для вычисления PageRank с использованием упрощенного Pregel подхода"""

    def __init__(self, db: Database):
        self.db = db
        self.damping_factor = PAGERANK_CONFIG['damping_factor']
        self.max_iterations = PAGERANK_CONFIG['max_iterations']
        self.tolerance = PAGERANK_CONFIG['tolerance']

        # Получение информации о документах
        self.documents = self.db.get_all_documents()
        self.doc_ids = [doc[0] for doc in self.documents]
        self.num_documents = len(self.doc_ids)

        if self.num_documents == 0:
            logger.warning("No documents in database")
            self.graph = None
            return

        # Создание отображения ID документа -> индекс графа
        self.id_to_index = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        self.index_to_id = {i: doc_id for i, doc_id in enumerate(self.doc_ids)}

        # Инициализация графа
        self.graph = SimplePregelGraph(self.num_documents)

        # Построение графа ссылок
        self.build_graph()

        logger.info(f"PageRankPregel initialized for {self.num_documents} documents")

    def build_graph(self):
        """Построение графа из данных базы"""
        for doc_id in self.doc_ids:
            source_idx = self.id_to_index[doc_id]
            outgoing = self.db.get_outgoing_links(doc_id)

            for target_doc_id, _ in outgoing:
                if target_doc_id and target_doc_id in self.id_to_index:
                    target_idx = self.id_to_index[target_doc_id]
                    self.graph.add_edge(source_idx, target_idx)

    def calculate(self) -> Dict[int, float]:
        """
        Основной метод вычисления PageRank
        """
        if not self.graph or self.num_documents == 0:
            logger.error("Cannot calculate PageRank: no documents or graph not initialized")
            return {}

        logger.info("Starting PageRank calculation using simplified Pregel approach")

        # Инициализация PageRank
        self.graph.initialize_pagerank()

        # Выполнение суперстепов
        for iteration in range(self.max_iterations):
            active = self.graph.run_superstep(self.damping_factor)

            # Сохранение текущих значений в базу данных
            current_pagerank = self.get_current_pagerank()
            for doc_id, rank in current_pagerank.items():
                self.db.update_pagerank(doc_id, rank, iteration + 1)

            logger.info(f"Superstep {iteration + 1}: {self.count_active_vertices()} active vertices")

            # Проверка условия остановки
            if not active:
                logger.info(f"PageRank converged after {iteration + 1} supersteps")
                break

        return self.get_current_pagerank()

    def get_current_pagerank(self) -> Dict[int, float]:
        """Получение текущих значений PageRank"""
        if not self.graph:
            return {}

        pagerank = {}
        for i, vertex in enumerate(self.graph.vertices):
            doc_id = self.index_to_id[i]
            pagerank[doc_id] = vertex.value

        return pagerank

    def count_active_vertices(self) -> int:
        """Подсчет активных вершин"""
        if not self.graph:
            return 0

        count = 0
        for vertex in self.graph.vertices:
            if vertex.active:
                count += 1
        return count

    def get_top_documents(self, n: int = 10) -> List[Tuple[int, float, str]]:
        """
        Получение топ-N документов по PageRank
        """
        pagerank = self.get_current_pagerank()

        if not pagerank:
            return []

        # Получение информации о документах
        documents_info = {}
        for doc_id, url, title in self.documents:
            documents_info[doc_id] = (url, title)

        # Сортировка по PageRank
        sorted_docs = sorted(pagerank.items(),
                             key=lambda x: x[1],
                             reverse=True)[:n]

        # Формирование результата
        result = []
        for doc_id, rank in sorted_docs:
            url, title = documents_info.get(doc_id, ("Unknown", "Unknown"))
            result.append((doc_id, rank, url, title[:50] + "..." if len(title) > 50 else title))

        return result

    def print_statistics(self):
        """Вывод статистики PageRank"""
        print("\n=== PageRank Statistics (Pregel) ===")
        print(f"Total documents: {self.num_documents}")
        print(f"Damping factor: {self.damping_factor}")
        print(f"Max supersteps: {self.max_iterations}")

        if self.graph:
            print(f"Current superstep: {self.graph.superstep}")
            print(f"Active vertices: {self.count_active_vertices()}")

        top_docs = self.get_top_documents(5)
        if top_docs:
            print("\nTop 5 documents by PageRank:")
            for doc_id, rank, url, title in top_docs:
                print(f"  ID: {doc_id}, Rank: {rank:.6f}")
                print(f"  URL: {url}")
                print(f"  Title: {title}")
                print()
        else:
            print("\nNo documents with PageRank calculated.")