from google.cloud import bigquery

def create_client():
    """Create the big query client

    Args:

    Returns:
        None
    """
    client = bigquery.Client('AbaloneML')
    hn_dataset_ref = client.dataset('hacker_news',
                                    project='bigquery-public-data')

