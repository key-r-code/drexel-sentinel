from src.engine.tools import add_to_calendar

result = add_to_calendar(
    title="TEST: Debug Event",
    date_str="2024-02-04",
    time_str=None,
    location="Bossone Research Enterprise Center",
    description="Room 201 - This is a test"
)

print(result)