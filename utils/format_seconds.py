def fmt_secs(secs, epochs):
    total_secs = round(secs * epochs)
    mins, secs = divmod(total_secs, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs