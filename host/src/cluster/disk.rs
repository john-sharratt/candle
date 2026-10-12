//! The volume a substrate is on, measured for fault mode
//! (`docs/replicated_substrate.md` §7.7): its free and total bytes, from the
//! mount whose point is the longest prefix of the substrate's directory.

use std::path::{absolute, Component, Path, PathBuf, Prefix};

use sysinfo::Disks;

/// A mounted volume: where, and its free and total bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mount {
    pub point: PathBuf,
    pub free: u64,
    pub total: u64,
}

/// The mount `dir` is on: the one whose point is the longest prefix of it.
/// A verbatim `dir` (`\\?\D:\x`, how a canonicalized path is spelt on
/// Windows) is matched as its plain spelling, the one mounts are listed in.
pub fn mount_of<'a>(dir: &Path, mounts: &'a [Mount]) -> Option<&'a Mount> {
    let dir = plain(dir);
    mounts
        .iter()
        .filter(|m| dir.starts_with(plain(&m.point)))
        .max_by_key(|m| m.point.components().count())
}

/// `path` without a Windows verbatim prefix: `\\?\D:\x` is `D:\x`, and
/// `\\?\UNC\server\share\x` is `\\server\share\x`. Anything else is unchanged.
fn plain(path: &Path) -> PathBuf {
    let mut parts = path.components();
    let Some(Component::Prefix(prefix)) = parts.next() else {
        return path.to_path_buf();
    };
    let head = match prefix.kind() {
        Prefix::VerbatimDisk(drive) => format!("{}:", drive as char),
        Prefix::VerbatimUNC(server, share) => format!(
            r"\\{}\{}",
            server.to_string_lossy(),
            share.to_string_lossy()
        ),
        _ => return path.to_path_buf(),
    };
    let mut out = PathBuf::from(head);
    out.push(parts.as_path());
    out
}

/// `dir`'s volume as `(free, total)`, measured now; `None` when no mount
/// holds it.
pub fn volume_of(dir: &Path) -> Option<(u64, u64)> {
    let dir = absolute(dir).ok()?;
    let mounts: Vec<Mount> = Disks::new_with_refreshed_list()
        .iter()
        .map(|d| Mount {
            point: d.mount_point().to_path_buf(),
            free: d.available_space(),
            total: d.total_space(),
        })
        .collect();
    mount_of(&dir, &mounts).map(|m| (m.free, m.total))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mount(point: &str, free: u64) -> Mount {
        Mount {
            point: PathBuf::from(point),
            free,
            total: 1000,
        }
    }

    /// The deepest mount holding the directory wins over the root it sits on.
    #[test]
    fn the_longest_mount_point_holding_the_dir_is_its_volume() {
        let mounts = [mount("/", 1), mount("/data", 2), mount("/data/sub", 3)];
        assert_eq!(
            mount_of(Path::new("/data/substrate"), &mounts)
                .unwrap()
                .free,
            2
        );
        assert_eq!(mount_of(Path::new("/data/sub/x"), &mounts).unwrap().free, 3);
        assert_eq!(mount_of(Path::new("/home/x"), &mounts).unwrap().free, 1);
        // `/data/sub` is a mount of its own, not a prefix of `/data/subway`.
        assert_eq!(
            mount_of(Path::new("/data/subway"), &mounts).unwrap().free,
            2
        );
        assert_eq!(mount_of(Path::new("/x"), &mounts[1..]), None);
    }

    /// The directory a test runs in is on some measured volume.
    #[test]
    fn the_working_directory_has_a_volume() {
        let (free, total) = volume_of(Path::new(".")).unwrap();
        assert!(total > 0 && free <= total);
    }

    /// **A canonicalized substrate is on its drive.** zend's workspace is
    /// canonicalized, so its substrate is spelt `\\?\D:\prog\substrate`, and
    /// matched verbatim against `D:\` it was on no mount — measured as 0 bytes
    /// free, which put the node in fault mode.
    #[cfg(windows)]
    #[test]
    fn a_verbatim_dir_is_on_its_plain_mount() {
        use std::fs::canonicalize;

        assert_eq!(
            plain(Path::new(r"\\?\D:\prog\substrate")),
            Path::new(r"D:\prog\substrate")
        );
        assert_eq!(
            plain(Path::new(r"\\?\UNC\nas\share\substrate")),
            Path::new(r"\\nas\share\substrate")
        );
        assert_eq!(plain(Path::new(r"D:\prog")), Path::new(r"D:\prog"));
        let mounts = [
            mount(r"C:\", 1),
            mount(r"D:\", 2),
            mount(r"\\nas\share\", 3),
        ];
        assert_eq!(
            mount_of(Path::new(r"\\?\D:\prog\substrate"), &mounts)
                .unwrap()
                .free,
            2
        );
        assert_eq!(
            mount_of(Path::new(r"\\?\UNC\nas\share\substrate"), &mounts)
                .unwrap()
                .free,
            3
        );
        assert_eq!(
            volume_of(&canonicalize(".").unwrap()).map(|(_, total)| total > 0),
            Some(true)
        );
    }
}
