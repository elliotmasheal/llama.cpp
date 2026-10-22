$ErrorActionPreference = 'SilentlyContinue'
$targets = @(
  'PCI\VEN_1002&DEV_1479&SUBSYS_14791002&REV_00\5&35cae7fa&0&000009',
  'PCI\VEN_1002&DEV_1681&SUBSYS_1DEC1043&REV_C8\4&31F4EC98&0&0041'
)
foreach ($t in $targets) {
  Write-Output "=== $t"
  $p = Get-PnpDeviceProperty -InstanceId $t
  foreach ($k in 'DEVPKEY_Device_FriendlyName','DEVPKEY_Device_LocationInfo','DEVPKEY_PciDevice_CurrentLinkSpeed','DEVPKEY_PciDevice_CurrentLinkWidth','DEVPKEY_PciDevice_MaxLinkSpeed','DEVPKEY_PciDevice_MaxLinkWidth','DEVPKEY_PciDevice_CurrentPayloadSize','DEVPKEY_PciDevice_MaxPayloadSize','DEVPKEY_Device_Parent') {
    $v = ($p | Where-Object { $_.KeyName -eq $k }).Data
    Write-Output ("  {0} = {1}" -f $k, $v)
  }
}
