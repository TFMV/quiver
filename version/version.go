package version

var (
	Version   = "dev"
	BuildDate = "02-24-2025"
)

func GetVersion() string {
	return Version
}

func GetBuildDate() string {
	return BuildDate
}
