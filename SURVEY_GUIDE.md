# Collecting real data in one day (Google Forms)

Make a form with these questions (one response = one trip the person remembers):
1. How far was the trip, roughly, in km? (number)
2. What hour did you start? (0-23)
3. Weekday or weekend? 
4. Weather: clear / rain / hot
5. Traffic: low / medium / high
6. Your budget for that trip: low / medium / high
7. How many people travelled together? (number)
8. Were you carrying luggage? yes / no
9. Which mode did you use? bus / metro / auto / cab / bike

Ask each respondent to fill it 2-3 times for different trips. Aim for 200+ rows.
Export responses to CSV, rename the columns to match the README table, convert yes/no and
weekday/weekend to 0/1, and save as `data/trips.csv`.
Mention in the paper that data is self-reported and how many people responded.
