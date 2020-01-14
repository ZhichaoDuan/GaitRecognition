def fmt_secs(secs, times):
    total_secs = round(secs * times)
    mins, secs = divmod(total_secs, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs