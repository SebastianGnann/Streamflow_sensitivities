def adjust_to_water_year_month(doy, start_month):
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    offset = sum(days_in_month[:start_month - 1])
    return ((doy - offset - 1) % 365) + 1

def get_water_year_string(month_num):
    # List of month abbreviations, 1-based index (Jan = 1)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Adjust for zero-based indexing
    month_str = months[month_num - 2]
    # Format as "A-Mon"
    wy_string = f"A-{month_str}"
    return wy_string
