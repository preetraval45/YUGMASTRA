"use client"

import { useState, useEffect } from 'react'
import { Menu, X, ChevronDown } from 'lucide-react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

interface NavLink {
  label: string
  href: string
  children?: NavLink[]
}

const navLinks: NavLink[] = [
  { label: 'Dashboard', href: '/dashboard' },
  { label: 'Attack Simulator', href: '/attack-simulator' },
  { label: 'Live Battle', href: '/live-battle' },
  {
    label: 'AI Agents',
    href: '#',
    children: [
      { label: 'Red Team', href: '/attacks' },
      { label: 'Blue Team', href: '/defenses' },
      { label: 'Threat Intel', href: '/threat-intelligence' },
      { label: 'Evolution', href: '/evolution' },
    ]
  },
  { label: 'Cyber Range', href: '/cyber-range' },
  { label: 'Analytics', href: '/analytics' },
  { label: 'Knowledge Graph', href: '/knowledge-graph' },
  { label: 'Settings', href: '/settings' },
]

export default function ResponsiveLayout({ children }: { children: React.ReactNode }) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const [openDropdown, setOpenDropdown] = useState<string | null>(null)
  const [isMobile, setIsMobile] = useState(false)
  const pathname = usePathname()

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768)
    }

    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  // Close mobile menu when route changes
  useEffect(() => {
    setIsMobileMenuOpen(false)
    setOpenDropdown(null)
  }, [pathname])

  const toggleDropdown = (label: string) => {
    setOpenDropdown(openDropdown === label ? null : label)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-black">
      {/* Mobile Header */}
      <header className="lg:hidden fixed top-0 left-0 right-0 z-50 bg-black/90 backdrop-blur-md border-b border-purple-500/20">
        <div className="flex items-center justify-between p-4">
          <Link href="/" className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
            YUGMĀSTRA
          </Link>
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="text-white p-2 hover:bg-purple-500/20 rounded-lg transition-colors"
            aria-label="Toggle menu"
          >
            {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <nav className="absolute top-full left-0 right-0 bg-black/95 backdrop-blur-md border-b border-purple-500/20 max-h-[calc(100vh-64px)] overflow-y-auto">
            <div className="p-4 space-y-2">
              {navLinks.map((link) => (
                <div key={link.label}>
                  {link.children ? (
                    <>
                      <button
                        onClick={() => toggleDropdown(link.label)}
                        className="w-full flex items-center justify-between px-4 py-3 text-white hover:bg-purple-500/20 rounded-lg transition-colors"
                      >
                        <span>{link.label}</span>
                        <ChevronDown
                          size={20}
                          className={`transform transition-transform ${
                            openDropdown === link.label ? 'rotate-180' : ''
                          }`}
                        />
                      </button>
                      {openDropdown === link.label && (
                        <div className="ml-4 mt-2 space-y-2">
                          {link.children.map((child) => (
                            <Link
                              key={child.href}
                              href={child.href}
                              className={`block px-4 py-2 rounded-lg transition-colors ${
                                pathname === child.href
                                  ? 'bg-purple-500/30 text-purple-300'
                                  : 'text-gray-300 hover:bg-purple-500/10'
                              }`}
                            >
                              {child.label}
                            </Link>
                          ))}
                        </div>
                      )}
                    </>
                  ) : (
                    <Link
                      href={link.href}
                      className={`block px-4 py-3 rounded-lg transition-colors ${
                        pathname === link.href
                          ? 'bg-purple-500/30 text-purple-300'
                          : 'text-white hover:bg-purple-500/20'
                      }`}
                    >
                      {link.label}
                    </Link>
                  )}
                </div>
              ))}
            </div>
          </nav>
        )}
      </header>

      {/* Desktop Sidebar */}
      <aside className="hidden lg:block fixed left-0 top-0 bottom-0 w-64 bg-black/50 backdrop-blur-md border-r border-purple-500/20 overflow-y-auto">
        <div className="p-6">
          <Link href="/" className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
            YUGMĀSTRA
          </Link>
        </div>

        <nav className="px-4 pb-4 space-y-2">
          {navLinks.map((link) => (
            <div key={link.label}>
              {link.children ? (
                <>
                  <button
                    onClick={() => toggleDropdown(link.label)}
                    className="w-full flex items-center justify-between px-4 py-3 text-white hover:bg-purple-500/20 rounded-lg transition-colors"
                  >
                    <span>{link.label}</span>
                    <ChevronDown
                      size={20}
                      className={`transform transition-transform ${
                        openDropdown === link.label ? 'rotate-180' : ''
                      }`}
                    />
                  </button>
                  {openDropdown === link.label && (
                    <div className="ml-4 mt-2 space-y-2">
                      {link.children.map((child) => (
                        <Link
                          key={child.href}
                          href={child.href}
                          className={`block px-4 py-2 rounded-lg transition-colors ${
                            pathname === child.href
                              ? 'bg-purple-500/30 text-purple-300'
                              : 'text-gray-300 hover:bg-purple-500/10'
                          }`}
                        >
                          {child.label}
                        </Link>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <Link
                  href={link.href}
                  className={`block px-4 py-3 rounded-lg transition-colors ${
                    pathname === link.href
                      ? 'bg-purple-500/30 text-purple-300'
                      : 'text-white hover:bg-purple-500/20'
                  }`}
                >
                  {link.label}
                </Link>
              )}
            </div>
          ))}
        </nav>
      </aside>

      {/* Main Content */}
      <main className="lg:ml-64 pt-16 lg:pt-0 min-h-screen">
        <div className="container mx-auto px-4 py-6 lg:px-8 lg:py-8">
          {children}
        </div>
      </main>
    </div>
  )
}
