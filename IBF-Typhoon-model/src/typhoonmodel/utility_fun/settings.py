import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


#API_LOGIN_URL = os.environ["API_LOGIN_URL"]
#API_SERVICE_URL = os.environ["API_SERVICE_URL"]
#IBF_API_URL=os.environ["IBF_API_URL"]

#ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")
##glofas


##datalake
#DATALAKE_STORAGE_ACCOUNT_NAME = os.environ["DATALAKE_STORAGE_ACCOUNT_NAME"]
#DATALAKE_STORAGE_ACCOUNT_KEY = os.environ["DATALAKE_STORAGE_ACCOUNT_KEY"]
#DATALAKE_API_VERSION = os.environ["DATALAKE_API_VERSION"]

az_credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
secret_client = SecretClient(vault_url='https://ibf-flood-keys.vault.azure.net', credential=az_credential)


ADMIN_LOGIN = secret_client.get_secret("ADMIN-LOGIN").value
PHP_PASSWORD=secret_client.get_secret("ETH-PASSWORD").value
UCL_USERNAME=secret_client.get_secret("UCL-USERNAME").value
UCL_PASSWORD=secret_client.get_secret("UCL-PASSWORD").value
AZURE_STORAGE_ACCOUNT=secret_client.get_secret("AZURE-STORAGE-ACCOUNT").value
AZURE_CONNECTING_STRING=secret_client.get_secret("AZURE-CONNECTING-STRING").value



# COUNTRY SETTINGS
SETTINGS_SECRET = {
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

