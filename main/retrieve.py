import os
from azure.cosmos import exceptions, CosmosClient, PartitionKey

endpoint = os.environ.get("AZURE_COSMOS_ENDPOINT", "")
key = os.environ.get("AZURE_COSMOS_KEY", "")
client = CosmosClient(endpoint, key)
database_name = 'data'
database = client.create_database_if_not_exists(id=database_name)
container_name = 'parser_data'
container = database.create_container_if_not_exists(
    id=container_name, 
    partition_key=PartitionKey(path="/url_hash"),
    offer_throughput=400
)
query = "SELECT * FROM c WHERE c.id = '589c5c94-0138-4918-9288-7d57244062e5'"
items = list(container.query_items(
    query=query,
    enable_cross_partition_query=True
))
request_charge = container.client_connection.last_response_headers['x-ms-request-charge']
print(items)
print('Query returned {0} items. Operation consumed {1} request units'.format(len(items), request_charge))