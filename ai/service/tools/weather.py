from ai.protocol.bot.tool_context import function_tool


@function_tool
async def get_weather(
    city: str,
    country: str | None = None,
) -> dict[str, str]:
    """
    Get the current weather for a given city and country.

    Args:
        city (str): The name of the city.
        country (str | None): The name of the country (optional).

    Returns:
        dict[str, str]: A dictionary containing the weather information.
    """
    # This is a placeholder implementation. Replace with actual weather API call.
    return {
        "city": city,
        "country": country or "Unknown",
        "temperature": "20°C",
        "condition": "Sunny",
    }
