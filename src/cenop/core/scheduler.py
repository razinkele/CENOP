"""
Scheduler for managing simulation time steps and scheduled tasks.

Translates scheduling logic from Repast Simphony to Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional
from enum import IntEnum


class Priority(IntEnum):
    """Task execution priorities (lower = earlier)."""
    
    FIRST = 0
    FOOD = 10
    DETERRENCE = 20
    PORPOISE = 30
    DAILY = 40
    MONTHLY = 50
    YEARLY = 60
    LAST = 100


@dataclass
class ScheduledTask:
    """A task scheduled to run at specific intervals."""
    
    action: Callable[[], None]
    start_tick: int = 0
    interval: int = 1
    priority: Priority = Priority.PORPOISE
    name: str = ""
    enabled: bool = True
    
    def should_run(self, tick: int) -> bool:
        """Check if this task should run at the given tick."""
        if not self.enabled:
            return False
        if tick < self.start_tick:
            return False
        if self.interval <= 0:
            return tick == self.start_tick
        return (tick - self.start_tick) % self.interval == 0


class Scheduler:
    """
    Manages scheduled tasks and their execution.
    
    Translates from: Repast Simphony ISchedule
    """
    
    def __init__(self):
        self._tasks: List[ScheduledTask] = []
        self._tick: int = 0
        
    def schedule_repeating(
        self,
        action: Callable[[], None],
        start: int = 0,
        interval: int = 1,
        priority: Priority = Priority.PORPOISE,
        name: str = ""
    ) -> ScheduledTask:
        """
        Schedule a task to run repeatedly.
        
        Args:
            action: Function to execute
            start: First tick to run
            interval: Ticks between executions
            priority: Execution priority
            name: Optional task name for debugging
            
        Returns:
            The scheduled task
        """
        task = ScheduledTask(
            action=action,
            start_tick=start,
            interval=interval,
            priority=priority,
            name=name
        )
        self._tasks.append(task)
        self._sort_tasks()
        return task
        
    def schedule_once(
        self,
        action: Callable[[], None],
        tick: int,
        priority: Priority = Priority.PORPOISE,
        name: str = ""
    ) -> ScheduledTask:
        """
        Schedule a task to run once at a specific tick.
        
        Args:
            action: Function to execute
            tick: Tick to run at
            priority: Execution priority
            name: Optional task name
            
        Returns:
            The scheduled task
        """
        task = ScheduledTask(
            action=action,
            start_tick=tick,
            interval=0,
            priority=priority,
            name=name
        )
        self._tasks.append(task)
        self._sort_tasks()
        return task
        
    def _sort_tasks(self) -> None:
        """Sort tasks by priority."""
        self._tasks.sort(key=lambda t: t.priority)
        
    def execute_tick(self, tick: int) -> None:
        """
        Execute all tasks scheduled for this tick.
        
        Args:
            tick: Current simulation tick
        """
        self._tick = tick
        
        for task in self._tasks:
            if task.should_run(tick):
                task.action()
                
        # Remove one-time tasks that have run
        self._tasks = [
            t for t in self._tasks
            if t.interval > 0 or t.start_tick > tick
        ]
        
    def remove_task(self, task: ScheduledTask) -> bool:
        """
        Remove a scheduled task.
        
        Args:
            task: Task to remove
            
        Returns:
            True if task was found and removed
        """
        if task in self._tasks:
            self._tasks.remove(task)
            return True
        return False
        
    def clear(self) -> None:
        """Remove all scheduled tasks."""
        self._tasks.clear()
        
    @property
    def current_tick(self) -> int:
        """Get the current tick."""
        return self._tick
