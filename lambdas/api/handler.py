# API Lambda handler
# Routes:
#   GET /variants/{id}                          → DynamoDB GetItem
#   GET /variants/search?q=...&field=...        → DynamoDB GSI Query (begins_with)
