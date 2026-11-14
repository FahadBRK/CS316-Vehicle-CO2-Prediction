# Dataset provenance

Natural Resources Canada, **2025 Fuel Consumption Ratings (2026-07-24)**.

- Official resource: https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64/resource/d589f2bc-9a85-4f65-be2f-20f17debfcb1
- Download: https://open.canada.ca/data/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64/resource/d589f2bc-9a85-4f65-be2f-20f17debfcb1/download/my2025-fuel-consumption-ratings.csv
- Retrieved: 14 November 2026.
- Original encoding: Windows-1252. The original CSV bytes are bundled unchanged.
- SHA-256: `46088cd4853987ef7a684f0b0af86e93e83dfaa378607a4750958c996248639c`.
- Actual snapshot: 701 records, 15 columns, model year 2025.
- Source documentation: https://open.canada.ca/data/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64/resource/ec44c747-a640-4481-bcf0-a2ea7ca8f249/download/understanding-the-tables.xlsx
- Licence: Open Government Licence - Canada, https://open.canada.ca/en/open-government-licence-canada

Contains information licensed under the Open Government Licence - Canada.
Natural Resources Canada does not endorse this student project.

This resource covers conventional vehicles and includes conventional hybrids.
Separate official resources cover battery-electric and plug-in hybrid vehicles.
Rows represent catalog vehicle/fuel configurations, not individual drivers or
sales-weighted fleet observations. CO2 is rated tailpipe g/km, not lifecycle CO2.

The resource is maintained and can change. The code intentionally verifies the
bundled snapshot hash instead of silently using updated data. Updating it requires
a fresh analysis and revision of the report and slides.
