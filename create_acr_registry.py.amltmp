from azure.identity import DefaultAzureCredential
from azure.mgmt.containerregistry import ContainerRegistryManagementClient

subscription_id = "c3b08fed-0f8b-4f46-bb93-74e6a86d2889"
resource_group = "akanuchibrightakb-rg"
registry_name = "akanuchiacr"

credential = DefaultAzureCredential()
client = ContainerRegistryManagementClient(credential, subscription_id)

poller = client.registries.begin_create(
    resource_group_name=resource_group,
    registry_name=registry_name,
    registry={
        "location": "eastus2",
        "sku": {"name": "Basic"},
        "admin_user_enabled": True
    }
)

result = poller.result()
print("Registry created:", result.login_server)
