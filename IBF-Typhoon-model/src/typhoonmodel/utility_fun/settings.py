import os

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


def get_settings(no_azure):
    if no_azure:
        PHP_PASSWORD = os.getenv("ETH_PASSWORD")
        UCL_USERNAME = os.getenv("UCL_USERNAME")
        UCL_PASSWORD = os.getenv("UCL_PASSWORD")
        AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
        AZURE_CONNECTING_STRING = os.getenv("AZURE_CONNECTING_STRING")
    else:
        az_credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
        secret_client = SecretClient(vault_url='https://ibf-flood-keys.vault.azure.net', credential=az_credential)
        PHP_PASSWORD=secret_client.get_secret("ETH-PASSWORD").value
        UCL_USERNAME=secret_client.get_secret("UCL-USERNAME").value
        UCL_PASSWORD=secret_client.get_secret("UCL-PASSWORD").value
        AZURE_STORAGE_ACCOUNT=secret_client.get_secret("AZURE-STORAGE-ACCOUNT").value
        AZURE_CONNECTING_STRING=secret_client.get_secret("AZURE-CONNECTING-STRING").value

    # COUNTRY SETTINGS
    return {
        "PHP": {
            "IBF_API_URL":'https://ibf-philippines.510.global/api/',
            "PASSWORD": PHP_PASSWORD,
            "UCL_USERNAME": UCL_USERNAME,
            "UCL_PASSWORD": UCL_PASSWORD,
            "AZURE_STORAGE_ACCOUNT": AZURE_STORAGE_ACCOUNT,
            "AZURE_CONNECTING_STRING": AZURE_CONNECTING_STRING,
            "mock": False,
            "if_mock_trigger": False,
            "notify_email": False
        },
    }

