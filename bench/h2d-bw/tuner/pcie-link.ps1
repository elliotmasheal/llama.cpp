$ErrorActionPreference = 'SilentlyContinue'
$out = foreach ($d in (Get-PnpDevice -Class PCI -Status OK)) {
  $g = { param($p) (Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName $p).Data }
  $cs = & $g 'DEVPKEY_PciDevice_CurrentLinkSpeed'
  $cw = & $g 'DEVPKEY_PciDevice_CurrentLinkWidth'
  $ms = & $g 'DEVPKEY_PciDevice_MaxLinkSpeed'
  $mw = & $g 'DEVPKEY_PciDevice_MaxLinkWidth'
  if ($null -ne $cw) {
    [PSCustomObject]@{
      Name     = $d.FriendlyName
      CurSpeed = $cs
      CurWidth = $cw
      MaxSpeed = $ms
      MaxWidth = $mw
      Instance = $d.InstanceId
    }
  }
}
$out | Format-Table -AutoSize -Wrap
