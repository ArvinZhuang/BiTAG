import arxivscraper
import os
if not os.path.exists('data'):
    os.makedirs('data')


def create_dataset(category, date_from, date_until):
    start_year = int(date_from.split("-")[0])
    start_month = int(date_from.split("-")[1])
    start_day = int(date_from.split("-")[2])

    end_year = int(date_until.split("-")[0])
    end_month = int(date_until.split("-")[1])
    end_day = int(date_until.split("-")[2])

    current_date = date_from
    if end_day <= start_day:
        next_month = start_month + 1
        if next_month > 12:
            next_month = 1
            next_year = start_year + 1
        else:
            next_year = start_year
        next_date = f"{next_year}-{next_month:02d}-{end_day:02d}"
    else:
        next_year = start_year
        next_month = start_month
        next_date = f"{next_year}-{next_month:02d}-{end_day:02d}"

    total_num_recodrs = 0
    while True:
        scraper = arxivscraper.Scraper(category=category, date_from=current_date, date_until=next_date)
        output = scraper.scrape()
        if output == 1:
            current_date = next_date
            if next_year == end_year and next_month == end_month:
                break
            next_month += 1
            if next_month % 13 == 0:
                next_year += 1
                next_month = 1
            next_date = f"{next_year}-{next_month:02d}-{end_day:02d}"
            continue

        total_num_recodrs += len(output)
        with open(f'{date_from}_to_{date_until}.tsv', 'a+') as file:
            for record in output:
                file.write(record['title'] + "\t" + record['abstract'] + '\n')
        print(f"Get record from {current_date} until {next_date}, total number of records: {total_num_recodrs}")

        current_date = next_date
        if next_year == end_year and next_month == end_month:
            break
        next_month += 1
        if next_month % 13 == 0:
            next_year += 1
            next_month = 1
        next_date = f"{next_year}-{next_month:02d}-{end_day:02d}"



if __name__ == "__main__":
    date_from = '2000-06-01'
    date_until = '2010-06-01'
    category = 'cs'
    create_dataset(category, date_from, date_until)
